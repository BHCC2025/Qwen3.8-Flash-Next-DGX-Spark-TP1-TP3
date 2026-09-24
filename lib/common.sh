# lib/common.sh — shared by recipes/tp{1,2,3}.sh. Sourced, not run.
# Builds the bash arrays each launcher splices into `docker run`. Every knob is an env var with a per-TP default
# that the launcher sets before calling the build_* functions.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -f "$REPO_DIR/cluster.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/cluster.env"
else
  echo "no $REPO_DIR/cluster.env — run ./setup.sh (or cp cluster.env.example cluster.env and edit it)" >&2; exit 2
fi

# NCCL network profiles (pair / triangle) live in the shared kit so ./setup.sh tests exactly these settings.
# shellcheck disable=SC1091
source "$REPO_DIR/kit/lib/nccl.sh"

IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
NAME="${NAME:-${CONTAINER_NAME:-vllm_qwen38fn}}"
PORT="${PORT:-8000}"
SERVED_NAMES="${SERVED_NAMES:-qwen3.8-flash-next}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/qwen38fn-vllm-cache}"
PLE_DIR="$REPO_DIR/patches/ple-offload"       # upstream patch set (see its PROVENANCE.md)
TP3_DIR="$REPO_DIR/patches/tp3-pad"           # ours
VP=/usr/local/lib/python3.12/dist-packages/vllm
NV=$VP/models/qwen4_exp/nvidia

# Variables forwarded from the head to the workers so every rank runs the same config.
FORWARD_VARS=(IMAGE NAME PLE_MODE PLE_WORKERS GRAPHS CAPTURE_SIZES KV_DTYPE OVERLAYS GMU MAXLEN SEQS MTP PORT MPORT CHUNK
              TOOL_PARSER DRAFT_VOCAB PREFIX_CACHE_ARG IB_GID_INDEX IB_GID_INDEX_TP2 NCCL_DEBUG NCCL_CHANNELS MTP_INDEX_SHARE ASYNC_SCHED
              LONGCTX EXTRA DOCKER_EXTRA)
forward_env() {
  local v out=""
  for v in "${FORWARD_VARS[@]}"; do [ -n "${!v+x}" ] && out+="$v=$(printf %q "${!v}") "; done
  printf '%s' "$out"
}

# Start rank N on a worker over SSH. The repo must exist at the same path there (./setup.sh copies it).
start_remote_rank() {  # host rank script
  local host=$1 rank=$2 script=$3
  echo "== rank $rank on $host"
  ssh -o BatchMode=yes "$host" "cd $(printf %q "$REPO_DIR") && $(forward_env) bash $(printf %q "$script") $rank" \
    || { echo "rank $rank on $host failed to start" >&2; exit 1; }
}

stop_on() {  # host...
  local h
  for h in "$@"; do ssh -n -o BatchMode=yes "$h" "docker rm -f $NAME" >/dev/null 2>&1 || true; done
}

# PLE_MODE: where the 47.7 GiB per-layer-embedding (n-gram) table lives.
#   none     = stock loader, table resident in unified memory (split across ranks at TP>1)
#   staged   = table on NVMe, rows gathered before each forward (decode CUDA graphs work) — TP1 default
#   mmap     = table on NVMe, gathered inside the forward (needs GRAPHS=piecewise|eager)
#   resident = our loader keeps each rank's slice as a plain tensor (TP>1)
#   offload  = vLLM's own VLLM_PLE_CPU_OFFLOAD
build_ple() {
  PLE_ENV=(); PLE_MOUNT=()
  case "$PLE_MODE" in
    none) ;;
    offload) PLE_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1) ;;
    mmap|staged|resident)
      PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}")
      PLE_MOUNT=(-v "$PLE_DIR/ple_layer.py:$NV/ple_layer.py:ro" -v "$PLE_DIR/ple_mmap.py:$NV/ops/ple_mmap.py:ro")
      [ "$PLE_MODE" = resident ] && PLE_ENV+=(-e QWEN4EXP_PLE_RESIDENT=1)
      if [ "$PLE_MODE" = staged ]; then
        PLE_ENV+=(-e QWEN4EXP_PLE_STAGED=1); PLE_MOUNT+=(-v "$PLE_DIR/model_state.py:$NV/model_state.py:ro")
      fi ;;
    *) echo "PLE_MODE must be none|staged|mmap|resident|offload" >&2; exit 2 ;;
  esac
}

# OVERLAYS=1: upstream vLLM fixes not in the pinned nightly + the MTP-loading fixes (patches/ple-offload/upstream-overlays)
build_overlays() {
  OVERLAY_MOUNT=()
  [ "${OVERLAYS:-1}" = 1 ] || return 0
  local o="$PLE_DIR/upstream-overlays"
  OVERLAY_MOUNT=(-v "$o/ops_ple.py:$NV/ops/ple.py:ro" -v "$o/ops_qsa.py:$NV/ops/qsa.py:ro" -v "$o/qsa.py:$NV/qsa.py:ro"
                 -v "$o/platforms_interface.py:$VP/platforms/interface.py:ro"
                 -v "$o/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro")
}

# GRAPHS: nocompile = decode CUDA graphs without torch.compile (default; compile duplicates a resident PLE table and
# has run Sparks out of memory), piecewise / full (with the compilation.py split-op patch), eager, default (vLLM's).
build_graphs() {
  GRAPH_ARGS=(); GRAPH_MOUNT=()
  case "$GRAPHS" in
    eager)     GRAPH_ARGS=(--enforce-eager) ;;
    piecewise) GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
               GRAPH_MOUNT=(-v "$PLE_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
    full)      GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
               GRAPH_MOUNT=(-v "$PLE_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
    nocompile) if [ -n "${CAPTURE_SIZES:-}" ]; then
                 GRAPH_ARGS=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}")
               else
                 GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
               fi ;;
    default)   ;;
    *) echo "GRAPHS must be nocompile|piecewise|full|eager|default" >&2; exit 2 ;;
  esac
}

# MTP speculative decoding (MTP=0 off). MTP_INDEX_SHARE=1 reuses the QSA indexer top-k across draft steps.
build_spec() {
  SPEC=()
  [ "${MTP:-0}" != 0 ] || return 0
  if [ "${MTP_INDEX_SHARE:-0}" = 1 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"index_share_for_mtp_iteration\":true}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
  fi
}

build_misc() {
  KV_ARGS=(); [ "${KV_DTYPE:-fp8_e4m3}" != auto ] && KV_ARGS=(--kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}")
  ASYNC_ARGS=(); [ "${ASYNC_SCHED:-0}" = 1 ] && ASYNC_ARGS=(--async-scheduling)
  CHUNK_ARGS=(); [ -n "${CHUNK:-}" ] && CHUNK_ARGS=(--max-num-batched-tokens "$CHUNK")
  # shellcheck disable=SC2206
  NAME_ARGS=(--served-model-name $SERVED_NAMES)
  # Standard GB10 (sm_121) vLLM environment. DeepGEMM faults on sm_121; FlashInfer autotune is disabled on the CLI.
  BASE_ENV=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a
            -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1
            -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1)
  # Serving flags common to every TP size. Thinking is off by default; turn it on per request with
  # chat_template_kwargs.enable_thinking=true. Prefix caching is off (GDN prefix-cache crash, vLLM #54173).
  SERVE_ARGS=(--host 0.0.0.0 --port "$PORT" --trust-remote-code --quantization modelopt
              --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" "${CHUNK_ARGS[@]}"
              --no-enable-flashinfer-autotune ${PREFIX_CACHE_ARG:---no-enable-prefix-caching}
              --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_xml}"
              --default-chat-template-kwargs '{"enable_thinking": false}')
}

build_all() { build_ple; build_overlays; build_graphs; build_spec; build_misc; }

check_model() {  # dir
  test -f "$1/config.json" || { echo "MODEL MISSING at $1 on $(hostname) — run ./setup.sh" >&2; exit 3; }
}

# DRY_RUN=1 prints the docker command instead of running it.
run_container() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'docker'; printf ' %q' "$@"; printf '\n'; return 0
  fi
  mkdir -p "$CACHE_DIR"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # Free page cache first: on a unified-memory box it counts against what the GPU can allocate.
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  docker "$@" >/dev/null
  sleep 3
  docker ps --format '{{.Names}} {{.Status}}' | grep "^$NAME " || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
}

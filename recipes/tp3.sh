#!/usr/bin/env bash
# recipes/tp3.sh — Qwen3.8-Flash-Next NVFP4 across THREE DGX Sparks (TP3), vLLM mp backend.
# Normally started via ./run.sh tp3 (or tp3-1m): workers rank 1 and 2 first, then the head (rank 0) serves the API.
#   recipes/tp3.sh 0|1|2    run one rank on the current node
#
# TP=3 needs padding: KV heads (2), GatedDeltaNet key heads (16) and the expert / shared-expert intermediate (640)
# do not divide by 3. patches/tp3-pad/tp_pad.py replicates KV heads 2->6 and GDN q/k heads 16->48 and zero-pads the
# expert intermediate 640->768 at load time; all three are exact (patches/tp3-pad/test_tp_pad.py). vLLM allocates the
# padded shapes from the edited config.json in MODEL_DIR_TP3 (scripts/prep-tp3-modeldir.sh). The vision tower
# (16 heads) is not padded: --mm-encoder-tp-mode data runs a full copy on each rank instead.
#
# Network (docs/networking.md): bootstrap over the LAN, data over both CX7 ports of each node, which in a triangle
# each reach ONE neighbour. MERGE_NICS=0 + SUBNET_AWARE_ROUTING=1 stop NCCL from merging the two ports and timing out
# (ibv_modify_qp RTR, 110) trying to reach a peer through the wrong one. P2P/SHM off, small NCCL buffers.
#
# LONGCTX=1 (./run.sh tp3-1m): 1M context via Qwen's static YaRN factor 4 (262,144 native). The YaRN lives in the
# config.json of MODEL_DIR_TP3_1M (prep-tp3-modeldir.sh --longctx), NOT in --hf-overrides: vLLM applies dict
# overrides to the target model only, the MTP draft stays at 262,144 and fails the mamba_block_size check.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
RANK="${1:?rank 0, 1 or 2}"

LONGCTX="${LONGCTX:-0}"
if [ "$LONGCTX" = 1 ]; then MAXLEN="${MAXLEN:-1000000}"; MODEL="$MODEL_DIR_TP3_1M"; else MODEL="$MODEL_DIR_TP3"; fi
PLE_MODE="${PLE_MODE:-none}"                   # table resident, 1/3 per rank (~16 GiB); PLE_MODE=mmap leaves it on disk
GMU="${GMU:-0.70}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MTP="${MTP:-3}"
CHUNK="${CHUNK-4096}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; GRAPHS="${GRAPHS:-nocompile}"; OVERLAYS="${OVERLAYS:-1}"
CAPTURE_SIZES="${CAPTURE_SIZES-}"
MPORT="${MPORT:-29533}"
[ "$GRAPHS" = full ] && { echo "GRAPHS=full is untested at TP3" >&2; exit 2; }
build_all

case "$RANK" in
  0) HEADLESS=() ;;
  1|2) HEADLESS=(--headless) ;;
  *) echo "rank must be 0, 1 or 2" >&2; exit 2 ;;
esac
HOST_IP="${LAN_IPS[$RANK]}"; HEAD_IP="${LAN_IPS[0]}"

check_model "$MODEL"
grep -q '"num_key_value_heads": 6' "$MODEL/config.json" \
  || { echo "$MODEL/config.json is not the TP3-padded one — run scripts/prep-tp3-modeldir.sh on $(hostname)" >&2; exit 3; }
LONG_ENV=()
if [ "$LONGCTX" = 1 ]; then
  grep -q '"rope_type": "yarn"' "$MODEL/config.json" \
    || { echo "$MODEL/config.json has no YaRN — run scripts/prep-tp3-modeldir.sh --longctx" >&2; exit 3; }
  LONG_ENV=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
fi

# TP3 overlay: tp_pad.py + model.py/mtp.py with the load hooks. This mtp.py already contains the reduced-vocab
# drafting code (built from mtp_draft_vocab.py), so DRAFT_VOCAB=N only needs the env var here.
TP3_MOUNT=(-v "$TP3_DIR/tp_pad.py:$NV/tp_pad.py:ro" -v "$TP3_DIR/model.py:$NV/model.py:ro" -v "$TP3_DIR/mtp.py:$NV/mtp.py:ro")
DRAFT_ENV=(); [ -n "${DRAFT_VOCAB:-}" ] && [ "$DRAFT_VOCAB" != 0 ] && DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")

nccl_env_triangle "$RANK"                     # kit/lib/nccl.sh: LAN bootstrap, both CX7 ports, no NIC merging

run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL:/models/qwen38fn:ro" -v "$CACHE_DIR:/root/.cache" \
  -e VLLM_HOST_IP="$HOST_IP" "${BASE_ENV[@]}" -e QWEN4EXP_TP_PAD=1 "${LONG_ENV[@]}" "${NCCL_ENV[@]}" \
  "${PLE_ENV[@]}" "${PLE_MOUNT[@]}" "${DRAFT_ENV[@]}" "${TP3_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${GRAPH_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn "${NAME_ARGS[@]}" "${SERVE_ARGS[@]}" --tensor-parallel-size 3 --mm-encoder-tp-mode data \
    "${SPEC[@]}" "${ASYNC_ARGS[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 3 --node-rank "$RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" "${HEADLESS[@]}" ${EXTRA:-}
echo "launched $NAME rank=$RANK host=$HOST_IP tp=3 longctx=$LONGCTX ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE mtp=$MTP gmu=$GMU maxlen=$MAXLEN"

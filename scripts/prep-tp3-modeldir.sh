#!/usr/bin/env bash
# scripts/prep-tp3-modeldir.sh — build MODEL_DIR_TP3 on every TP3 node (or MODEL_DIR_TP3_1M with --longctx).
#   scripts/prep-tp3-modeldir.sh [--longctx] [--local]     --local = this node only (what the fan-out runs)
# ./setup.sh runs both forms for a 3-node cluster. The new dir is hardlinks to every file of MODEL_DIR except
# config.json, which gets the TP3-padded sizes (see patches/tp3-pad/tp_pad.py for each one). No extra disk. Idempotent.
# --longctx also writes Qwen's static YaRN (factor YARN_FACTOR, default 4.0: 262,144 -> 1M) into that config.json.
# It must live there, not in --hf-overrides: vLLM applies dict overrides to the target only, so the MTP draft kept
# 262,144 and failed the mamba_block_size check.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
LONGCTX=0; LOCAL=0
for a in "$@"; do case "$a" in --longctx) LONGCTX=1 ;; --local) LOCAL=1 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac; done
if [ "$LOCAL" = 0 ]; then
  args=(--local); [ "$LONGCTX" = 1 ] && args+=(--longctx)
  bash "$REPO_DIR/scripts/prep-tp3-modeldir.sh" "${args[@]}"
  for w in "${NODES[@]:1:2}"; do
    ssh -o BatchMode=yes "$w" "cd $(printf %q "$REPO_DIR") && YARN_FACTOR=${YARN_FACTOR:-4.0} bash scripts/prep-tp3-modeldir.sh ${args[*]}"
  done
  exit 0
fi
SRC="$MODEL_DIR"; YARN_FACTOR="${YARN_FACTOR:-4.0}"
if [ "$LONGCTX" = 1 ]; then DST="$MODEL_DIR_TP3_1M"; else DST="$MODEL_DIR_TP3"; fi
test -f "$SRC/config.json" || { echo "missing $SRC on $(hostname) — run ./setup.sh" >&2; exit 3; }
mkdir -p "$DST"
for f in "$SRC"/*; do b=$(basename "$f"); [ "$b" = config.json ] && continue; ln -f "$f" "$DST/$b"; done
python3 - "$SRC/config.json" "$DST/config.json" "$LONGCTX" "$YARN_FACTOR" <<"PY"
import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c)
want = {"num_key_value_heads": (2, 6), "linear_num_key_heads": (16, 48),
        "moe_intermediate_size": (640, 768), "shared_expert_intermediate_size": (640, 768)}
for k, (old, new) in want.items():
    assert t[k] == old, f"{k}={t[k]}, expected {old} (checkpoint changed?)"
    t[k] = new
if sys.argv[3] == "1":
    rp = t["rope_parameters"]
    assert rp.get("rope_type") == "default", rp
    rp.update(rope_type="yarn", factor=float(sys.argv[4]), original_max_position_embeddings=t["max_position_embeddings"])
    want["rope_parameters"] = None
    # Qwen4ExpConfig copies text rope_parameters to the top level unless the top level has its own, and transformers
    # then validates YaRN on the top-level config, which has no max_position_embeddings (AttributeError). Pin the top
    # level to the original default rope; the model reads RoPE from text_config only.
    top = {k: v for k, v in rp.items() if k not in ("factor", "original_max_position_embeddings")}
    top["rope_type"] = "default"
    c["rope_parameters"] = top
json.dump(c, open(sys.argv[2], "w"), indent=2)
print("config:", {k: t[k] for k in want})
PY
echo "$(hostname): $DST ready ($(ls "$DST" | wc -l) files)"

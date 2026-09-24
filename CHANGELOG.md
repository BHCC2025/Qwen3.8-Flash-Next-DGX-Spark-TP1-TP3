# Changelog

## 0.2.0 — 2026-09-24

- `./setup.sh` (shared dgx-spark-recipe-kit under `kit/`): nodes and SSH, dependencies, cabling detection, fabric
  IPs, `cluster.env`, image, RDMA + NCCL network test, model download and staging. Replaces
  `scripts/preflight.sh`, `download-model.sh` and `stage-nodes.sh`.
- NCCL settings moved to `kit/lib/nccl.sh`, so the setup test and the launchers use identical settings.
- Added `scripts/prep-tp3-modeldir.sh`, which 0.1.0 referenced but didn't include.
- TP1 and TP2 benched on our own Sparks with this repo's launcher. Every row is now verified.

## 0.1.0 — 2026-09-24

- First release: TP1, TP2, TP3 and TP3 at 1M context behind one `run.sh`, configured through `cluster.env`.
- TP3 load-time padding (`patches/tp3-pad/`) with exactness tests.
- 1M context via static YaRN in `config.json` (target and MTP draft).
- Vendored the upstream NVMe n-gram-table patch set (tonyd2wild, Apache-2.0), plus the one-line `FP8_PB_WO`
  alias that checkpoint revision `fc694b54` needs.
- Preflight, staging, smoke-test and benchmark scripts.
- TP3 and TP3-1M benched on our own Sparks. TP1 and TP2 still carry upstream figures until benched.

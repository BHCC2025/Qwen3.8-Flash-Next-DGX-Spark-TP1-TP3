# Changelog

## 0.3.1 — 2026-09-29

- **TP3 and TP3-1M benched with this repo's launcher and `bench/bench.sh`** (`bench/results/2026-09-29-tp3.md`). The
  earlier TP3 figures came from the pre-repo launcher and had no smoke-test log, although the README said every
  row was benched and passed it. Now every row is. TP3: 62.8 / 40.7 tok/s code / prose; TP3-1M needle test 3/3 up to
  988K. KV pool sizes are quoted from vLLM's log (2.38M tokens for TP3, 2.46M for TP3-1M).
- `kit/` updated to dgx-spark-recipe-kit v0.3.0: `./setup.sh --check` checks every node in `cluster.env` (it only
  checked the head), the RDMA test works for a second user, and the benchmark refuses to file another model's
  results here.
- Environment overrides of `MODEL_DIR`, `MODEL_DIR_TP3`, `MODEL_DIR_TP3_1M` and `CACHE_DIR` now reach the workers.
- `DRY_RUN=1` prints only the docker commands; `./run.sh` usage no longer prints code.
- Docs corrected: disk ~130 GB (what `./setup.sh` checks), `GRAPHS=full` listed, NOTICE lists the kit and says the
  TP2 NCCL settings come from it, padding doc no longer contradicts itself on embeddings, provenance tables carry
  sha256, the unused `*_e5m2.py` overlays are named as unused; GitHub issue template.
- Launch commands unchanged (checked with `DRY_RUN=1` against 0.3.0 for tp1, tp2, tp3 and tp3-1m).

## 0.3.0 — 2026-09-29

- `kit/` updated to dgx-spark-recipe-kit v0.2.1:
  - TP2 uses the faster pair NCCL profile (the triangle's buffer and protocol settings). Re-benched: decode after a
    ~9K-token prompt 33.4 → 39.0 tok/s, the rest unchanged within noise (`bench/results/2026-09-29-tp2.md`).
    TP1 and TP3 launch with identical docker commands, so their numbers stand.
  - `bench/bench.sh` and `scripts/smoke-test.sh` now run the kit's shared suite (same test code as before), so every
    recipe is measured the same way.
  - `./setup.sh --check` works on a fresh multi-node clone (it used to fail or stop in the network test).
- `cluster.env` values can be overridden from the environment for one run (`PORT=8001 ./run.sh tp1`); they used to
  be silently ignored.
- `DRY_RUN=1` prints the docker commands before the model is downloaded (it stopped at MODEL MISSING).
- `./run.sh status` checks the head locally instead of over SSH to itself.
- The shared bench: `LONG=1` includes the 988K needle test on a 1M server, so the published 988K result can be
  reproduced; a failed smoke test makes the bench exit non-zero; the long-prompt test is labelled ~9K tokens (not
  ~12K) here and in the README. (Also in this release: kit v0.1.1, the same code as before as a tagged release.)

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

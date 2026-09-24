# Troubleshooting

Start with `./setup.sh --check` (it changes nothing), then look at the server log (`./run.sh logs` on the head, `docker logs
vllm_qwen38fn` on a worker).

| Symptom | Cause | Fix |
|---|---|---|
| `ibv_modify_qp ... RTR ... 110` / NCCL timeout at TP3 | NCCL merged the two CX7 ports, or it's using the IPv6 GID | Check `IB_GID_INDEX` (docs/networking.md). The recipe already sets `MERGE_NICS=0`, `CROSS_NIC=1`, `SUBNET_AWARE_ROUTING=1` |
| Head waits forever for workers | Wrong `LAN_IPS`/`LAN_IF`, a firewall on `MPORT` (29531/29533), or a worker that exited | `docker logs vllm_qwen38fn` on each worker; `./run.sh stop` and start again |
| OOM or a Spark reboots while loading | Other containers are holding memory; page cache; `GRAPHS=default`/`piecewise` with the table in memory (torch.compile copies it) | Stop everything else. Keep `GRAPHS=nocompile` with `PLE_MODE=none`. Lower `GMU` |
| `MODEL MISSING` / `not the TP3-padded one` | The model or TP3 dir isn't on that node | `./setup.sh` (copies the model and builds the TP3 dirs) |
| `no YaRN` with `tp3-1m` | 1M dir not built | `scripts/prep-tp3-modeldir.sh --longctx` |
| `mamba_block_size` error with a custom long context | YaRN passed via `--hf-overrides` | Use the `--longctx` model dir (docs/long-context.md) |
| TP1 decode slow, disk busy | The n-gram table is read from NVMe per token; another process is using the disk | Keep the model on local NVMe (not NFS) and nothing else heavy on that disk |
| Tool calls come back as text | Wrong parser | `TOOL_PARSER=qwen3_xml` (default) or `qwen3_coder` |
| Crash with prefix caching on | GDN prefix-cache bug, vLLM #54173 | Leave prefix caching off (the default) |
| DeepGEMM / FlashInfer autotune errors | Not supported on sm_121 in this image | The recipe already sets `VLLM_USE_DEEP_GEMM=0` and `--no-enable-flashinfer-autotune` |

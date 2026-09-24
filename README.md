# Qwen3.8-Flash-Next on 1, 2 or 3 DGX Sparks (TP1–TP3)

Run NVIDIA's `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint with vLLM on one, two or three NVIDIA DGX Sparks
(GB10, 128 GB unified memory each). One script, `./run.sh tp1|tp2|tp3`, and one config file describing your nodes.

The TP3 recipe is new here. The model's KV heads (2), GatedDeltaNet key heads (16) and expert width (640) don't
divide by 3, so stock vLLM can't split it across three Sparks. This repo pads those dimensions at load time. The
padding is exact (the tests prove it), and the result runs at up to 1M context.

| Sparks | Command | Context | Decode, single stream (code / prose) | Cold prefill 8K | Verified |
|---|---|---|---|---|---|
| 1 | `./run.sh tp1` | 256K | 40.1 / 26.4 tok/s | 1,779 tok/s | 2026-09-24 |
| 2 | `./run.sh tp2` | 256K | 51.7 / 37.6 tok/s | 2,882 tok/s | 2026-09-24 |
| 3 | `./run.sh tp3` | 256K | **56.5 / 39.3 tok/s** | 2,669 tok/s | 2026-09-24 |
| 3 | `./run.sh tp3-1m` | **1M** | 57.4 / 41.1 tok/s | ~1,900 tok/s at 988K | 2026-09-24, needle test 3/3 up to 988K |

Every row was benched on our own Sparks with `bench/bench.sh` (same prompts for every row) and passed the smoke test.
See [bench/results/](bench/results/).

- **Endpoint:** `http://<head>:8000/v1` (OpenAI-compatible), model `qwen3.8-flash-next`
- **Defaults:** thinking off (turn it on per request with `"chat_template_kwargs": {"enable_thinking": true}`),
  tool calling on (`qwen3_xml` parser), MTP speculative decoding with 3 tokens, FP8 KV cache, 6 concurrent sequences.

## Requirements

| | |
|---|---|
| Hardware | 1–3 DGX Spark (or other GB10 boxes with a ConnectX-7) |
| Cables | TP2: one QSFP cable. TP3: three, in a triangle (see [docs/networking.md](docs/networking.md)) |
| OS | DGX OS 7 (Ubuntu 24.04), Docker with the NVIDIA runtime |
| Disk | ~124 GB free NVMe on **every** node (each node needs its own local copy of the model) |
| Image | `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (pinned; the patches are made for it) |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` |
| Access | SSH from the head node to the workers (`setup.sh` sets up key login); `sudo` for installs and fabric IPs |

## Quick start

On the Spark you'll serve from (the head node):

```bash
git clone https://github.com/BHCC2025/Qwen3.8-Flash-Next-DGX-Spark-TP1-TP3.git
cd Qwen3.8-Flash-Next-DGX-Spark-TP1-TP3
./setup.sh
```

`setup.sh` asks how many Sparks you have (1, 2 or 3) and their SSH names, then:
- checks and installs what's missing
- works out your cabling and assigns fabric IPs if the ports have none, after asking
- writes `cluster.env` for you
- pulls the image and **tests the network with a real NCCL all-reduce** before anything big is downloaded
- downloads the model once, copies it to the other Sparks, and builds the TP3 model folders

It asks before every change. Re-run it any time. `./setup.sh --check` only reports.

Then start it:

```bash
./run.sh tp1        # 1 Spark
./run.sh tp2        # 2 Sparks, one QSFP cable between them
./run.sh tp3        # 3 Sparks, QSFP triangle (each Spark cabled to both others)
./run.sh tp3-1m     # 3 Sparks, 1M context
./run.sh status     # wait for "serving: [...]"; loading takes several minutes
scripts/smoke-test.sh
```

Stop with `./run.sh stop`, which stops the container on every node listed in `cluster.env`. Cabling details are in
[docs/networking.md](docs/networking.md).

## Settings

Set any of these in the environment for one run (`MTP=4 SEQS=8 ./run.sh tp2`). `DRY_RUN=1` prints the docker
commands and starts nothing.

| Variable | TP1 | TP2 | TP3 | What it does |
|---|---|---|---|---|
| `PLE_MODE` | `staged` | `none` | `none` | Where the 47.7 GiB n-gram table lives: `none` = in memory (split across ranks), `staged`/`mmap` = on NVMe, rows read on demand |
| `MTP` | 3 | 3 | 3 | Speculative tokens per step (0 = off) |
| `DRAFT_VOCAB` | 65536 | off | off | Reduced-vocab MTP draft |
| `SEQS` | 6 | 6 | 6 | Max concurrent sequences |
| `MAXLEN` | 262144 | 262144 | 262144 (1M with `tp3-1m`) | Max context |
| `GMU` | 0.80 | 0.70 | 0.70 | vLLM `--gpu-memory-utilization` |
| `CHUNK` | 4096 | 4096 | 4096 | `--max-num-batched-tokens` (the biggest speed lever on GB10) |
| `KV_DTYPE` | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 | `auto` = BF16 |
| `GRAPHS` | nocompile | nocompile | nocompile | CUDA-graph mode: `nocompile`, `piecewise`, `eager`, `default` |
| `TOOL_PARSER` | qwen3_xml | qwen3_xml | qwen3_xml | or `qwen3_coder` |
| `EXTRA` / `DOCKER_EXTRA` | | | | extra args for vLLM / `docker run` |

The recipe headers in [recipes/](recipes/) list the rest.

## How it works

- **TP1:** the weights take ~76 GiB and the per-layer-embedding (n-gram) table another 47.7 GiB. That leaves no
  room for KV cache, so the table stays on NVMe and its rows are read with `preadv` just before each forward
  pass. This is the upstream single-Spark patch set, vendored in [patches/ple-offload/](patches/ple-offload/).
- **TP2:** each rank holds half the table in memory. NCCL talks over RoCE on the single cable.
- **TP3:** see [docs/tp3-padding.md](docs/tp3-padding.md). Load-time padding (`patches/tp3-pad/`) plus a
  `config.json` edited to the padded sizes. NCCL bootstraps over the LAN and sends data over both CX7 ports.
- **1M context:** see [docs/long-context.md](docs/long-context.md). Qwen's static YaRN factor 4 is written into
  `config.json` so the MTP draft model picks it up too.

## Benchmarks

`bench/bench.sh LABEL` runs the same suite against whatever is serving on `:8000`:
- single-stream decode for code, prose and a ~12K-token prompt
- cold prefill at 8K and 28K tokens with unique prompts (prefix cache off)
- the smoke test; add `LONG=1` for the needle test

Results and raw logs go in [bench/results/](bench/results/).

## Troubleshooting

Run `./setup.sh --check` and read the FAIL lines. It writes `.setup/report.txt`, which is what to attach to an issue.
See also [docs/troubleshooting.md](docs/troubleshooting.md). The two most common problems:
- NCCL `ibv_modify_qp ... 110` at TP3: the wrong GID index or merged NICs.
- Out of memory while loading: other containers are still running, or page cache is taking memory. Run `./run.sh stop` on everything and check `free -g`.

## Credits

The TP1 and TP2 recipes and the NVMe n-gram-table patch set come from
[tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
(Apache-2.0). That work vendors upstream vLLM PRs by peakcrosser7 (#55375) and andreasgru (#54846).

The TP3 NCCL settings for the RoCE triangle follow the
[DeepSeek-V4.1 3-Spark stack by MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks); only the
settings were used, no code. vLLM is Copyright contributors to the vLLM project, Apache-2.0.

Full details are in [NOTICE](NOTICE) and each patch directory's `PROVENANCE.md`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

# Qwen3.8 Flash Next on Strix Halo 128 GB

This fork reconstructs **checkpoint 004**, a measured llama.cpp configuration for Qwen3.8 Flash Next on a Ryzen AI Max+ PRO 395 with Radeon 8060S and 128 GB RAM. It uses Vulkan, the original Unsloth UD-Q4_K_XL target weights, F16 target and draft KV caches, a 210k context, and two-token shared-head MTP.

The purpose is to share a reproducible configuration and its source changes. Validation covers this model and machine. Other models, devices, operating systems and concurrent serving configurations have not been qualified.

**Known limitation:** repeated requests can produce different full probabilities when old cache payloads are reused. This behavior was present in earlier checkpoints and remains in 004. The exact quality comparisons below clear cache data between test windows. They do not establish stable probabilities for every serving history.

AI agents implemented and integrated local changes, wrote tests and documentation, and reviewed the work. Those reviews were not external human peer review. Borrowed code is credited in the commits and [patch guide](PATCHES.md). This repository does not claim upstream approval or a novel algorithm.

## Start from the tested source

Use branch `strix-checkpoint-004` or tag `checkpoint-004`. The history starts at the [upstream revision used for the measurements](https://github.com/ggml-org/llama.cpp/commit/2145525a4081d66ff1a87cf43ef809f95a85ac0c). Updating that base creates a new configuration that needs its own qualification.

The 18 source commits were reconstructed from the saved patches, in dependency order. At the end of each checkpoint stage, the complete Git source tree matched that checkpoint's frozen source. Individual intermediate commits were not rebuilt or separately qualified. A final documentation commit adds this guide and a portable launcher without changing the inference implementation.

The [patch guide](PATCHES.md) explains what each group does and where it came from. Upstream history, licensing and copyright notices are retained.

## Build on the target machine

The tested environment was Debian 13, GCC 14.2 and RADV 26.1.6. Use the official distribution repositories for packages. The container needs access to the host's Vulkan render device.

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build git python3 \
    glslc libvulkan-dev spirv-headers libssl-dev \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libgomp1 libstdc++6 libgcc-s1 libssl3t64 zlib1g libzstd1

cmake -S . -B build-strix -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=ON
cmake --build build-strix --parallel 16 --target \
    llama-server llama-bench test-backend-ops test-alloc test-recurrent-state-rollback
```

Run these commands from the repository root. `GGML_NATIVE=ON` compiles for the build machine's CPU. Build on the intended Strix host.

## Obtain the matching weights

Weights are external and are not included in this repository.

- Download all four files in [Unsloth's tested UD-Q4_K_XL directory](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL). Keep the shards together.
- Download the [matching shared Q8 MTP head](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf).

The Q8 MTP file contains draft weights. It does not replace the target's IQ4_NL PLE tensor or upgrade target precision.

## Run

Stop another model server using the GPU before starting this configuration. From the repository root:

```sh
export STRIX_MODEL=/path/to/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export STRIX_DRAFT=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./examples/strix/run-server.sh
```

The launcher uses the binary in `build-strix/bin`. `STRIX_SERVER` can select a different executable. `STRIX_MTP=0` disables drafting. Append server arguments to override defaults, for example:

```sh
./examples/strix/run-server.sh --host 127.0.0.1 --port 7001
```

The default is port 7000 on all interfaces. Other defaults match checkpoint 004: all GPU layers, Vulkan0, flash attention, mmap, lazy PLE reads, batch and microbatch 2048, 16 CPU threads, one slot, F16 K/V and context 210000. Sampling uses temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repeat penalty 1. Thinking is preserved with `reasoning_effort` set to `xhigh`.

Two flags enable the local optimizations:

```sh
GGML_VK_MMQ_ID_COMPACT=1
LLAMA_QWEN4EXP_MTP_CACHE_ONLY=1
```

The launcher sets both. Keep these settings fixed throughout a context's lifetime. MTP proposes at most two tokens for target verification. Target and draft caches both remain F16.

## Measured performance

PP means prompt processing and TG means generation. Rates are tokens per second.

| Configuration | 1,024-token PP | 1,024-token TG | 209,000-token PP | 209,000-token TG |
| --- | ---: | ---: | ---: | ---: |
| Official b11179, original settings | 323.03 | 23.73 | 154.62 | 7.58 |
| Checkpoint 001: integrated patches and MTP | 457.02 | 38.10 | 300.15 | 30.46 |
| Checkpoint 002: fewer copies | 457.85 | 38.54 | 361.03 | 30.28 |
| Checkpoint 003: compact expert dispatch | 479.52 | 38.99 | 363.60 | 30.55 |
| **Checkpoint 004: MTP cache-only catch-up** | **475.63** | **38.64** | **409.55** | **31.00** |

Checkpoint 004 processed the 209k prompt in 510.31 seconds, compared with 574.80 seconds for 003. That is **12.64% higher prompt throughput and 64.49 seconds saved**. Short PP and TG measured 0.8-0.9% lower than 003. The long-prompt gain is the reason for this release. A short-context gain or absence of every possible slowdown is not established.

The original official baseline used microbatch 512. The checkpoints use 2048. The table compares complete configurations and does not attribute each improvement to an individual patch.

Each measurement used the same saved input token IDs, 128 output tokens, temperature 1, seed 12345 and disabled prompt reuse. Previous results were reused rather than rerun. No other compute ran alongside the measurements. Filesystem cache, temperature and power conditions were uncontrolled. These are single-run results, so small differences do not establish repeatable gains.

The 004 long run enabled activation logging to prove the new path executed. Its short result above used release defaults with logging off. Both 128-token continuations and MTP acceptance counts matched 003 exactly. Peak sampled GTT use was 99.15 GiB, with no OOM or memory-limit counter increases.

## What the quality checks establish

Across 4,352 gold-token positions at short, 32k and 208k histories, all 248,320 logits per position were byte-identical to the saved checkpoint 001-003 references. All 52 repeated control rows matched the 104 saved controls. KL divergence and gold NLL difference were zero. Conditional perplexity was identical on these tested, cleared histories. [Machine-readable summary](validation.json).

These comparisons use earlier checkpoints as references. They are not a new full-distribution comparison against the official b11179 baseline or BF16 weights.

Separate MTP probes compare 13 complete K/V and logical-state snapshots and seven subsequent hidden and raw top-k output rows. CPU, Vulkan and real-weight checks passed. Four eligible decodes must prove activation, including graph reuse, suffix rejection and replacement. The real-weight probe uses synthetic target hidden inputs with the server's top-k sampler. Serving also passed the saved greedy repetition and cached/cold continuation checks.

The target-logit test does not exercise drafting. The MTP state probes and serving checks cover that path separately. Matching finite test inputs is not proof of unchanged quality for every request. The cache-reuse limitation at the top of this page remains unresolved.

The repository includes the test implementations and a compact result summary. It does not include the multi-gigabyte raw captures, frozen prompt datasets, local worklog or model weights. The summary records completed measurements and is not a self-contained reproduction of the full evaluation dataset.

## Scope of this fork

This is a source snapshot of a measured configuration, with the limitations above. There is no claim that it replaces upstream llama.cpp across its supported hardware and models. Publishing the fork does not submit these changes for upstream review. Any later upstream contribution must independently meet [llama.cpp's contribution rules](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md).

Keep the `checkpoint-004` tag fixed. Later work can use a new branch and a new numbered tag after qualification. No additional model run or benchmark was performed merely to reorganize this source history.

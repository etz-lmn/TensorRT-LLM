# adaptive-eplb-full: build and run

Branch `adaptive-eplb-full` on `git@github.com:etz-lmn/TensorRT-LLM.git`, based on upstream TensorRT-LLM
`1f3b31be8` (the commit of the `1.4.0rc5.dev202611190007` nightly). It carries, in order (after them: this file and a fix of the runner's kernel arguments):

| commit | content |
|---|---|
| `ac1faf36f` `4e25c4331` `27c0d6551` | MegaMoE integration (#22099): empty-batch gate fix, FC1-ready drain and proxy fences in the vendored token comm, one-launch input staging, padding-row drop, EPLB allreduce joined to the launch |
| `fc6638c43` | MTP checkpoint fix (#22096) |
| `b5f6c189d` `cb8db663f` `2527f16ab` | Adaptive EPLB (#22100): policy, runtime, transport, adaptive route in the MegaMoE router, on-GPU time-model calibration, tests |
| `3fb84c503` | the rest of the MegaMoE kernel changes of cutedsl_megamoe!173: Blackwell `token_tile` return readiness, adaptive-route guards in the dedup and Rubin local routers; the op no longer refuses `token_tile` on SM100/SM103 |

## Build

The branch changes C++ (`cpp/tensorrt_llm/kernels/moe/...`, `thop/moe/...`, `quantization.cu`), so the Python
package alone is not enough: build the wheel.

1. Start from a container that can build for GB300 (SM103), e.g. the release image the branch is based on:
   `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc5.dev202611190007` (CUDA 13.4, torch 2.14, nvidia-cutlass-dsl 4.8.0),
   or the repository's own dev image (`docker/Makefile`, `make -C docker build`).
2. In the checkout (with submodules and LFS):
   ```bash
   git clone -b adaptive-eplb-full https://github.com/etz-lmn/TensorRT-LLM.git && cd TensorRT-LLM
   git submodule update --init --recursive && git lfs pull
   python3 scripts/build_wheel.py -a "100;103" --clean
   pip install build/tensorrt_llm-*.whl
   ```
   `-a` takes plain SM numbers (no `-real` suffix). Left out, it builds every arch (about 2.4x the compile time).
   For GB300 keep `100` next to `103`: `103` alone drops the SM100 entries of the trtllm-gen kernel tables that
   GB300 also uses.
3. Check: `python3 -c "from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig"`.

## Enable adaptive EPLB

Requirements: `moe_config.backend: MEGAMOE_CUTEDSL`, expert parallelism (`moe_expert_parallel_size > 1`, attention
DP), all EP ranks in one NVLink domain (one node or one MNNVL rack), `num_slots` >= number of experts and divisible
by the EP size.

```yaml
enable_attention_dp: true
moe_expert_parallel_size: 32
moe_config:
  backend: MEGAMOE_CUTEDSL
  load_balancer:
    mode: adaptive
    num_slots: 512                # 384 experts + 4 replica slots per rank on 32 ranks
    time_model: auto              # default; measure on this deployment's GPUs during CUDA graph warmup
    calibration_cache_dir: /path/on/shared/disk/adaptive_eplb   # optional; default $TRTLLM_CACHE_DIR/adaptive_eplb
    # rows_per_expert: 512        # default statistics window
```

Python API (`from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig, MoeConfig`):
`moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL", load_balancer=MoeAdaptiveLoadBalancerConfig(num_slots=512))`.

- `time_model: auto`: the first start measures a calibration per decode CUDA-graph bucket (seconds per bucket) and
  writes it under `calibration_cache_dir` (rank 0); later starts with the same GPU, EP group, slots, kernels and
  buckets reuse it. `time_model: file` + `time_model_file` uses a calibration written offline by
  `examples/wide_ep/ep_load_balancer/measure_adaptive_eplb_time_model.py`. When no calibration holds, the log says
  why and the layers serve their home layout (no replicas), i.e. behaviour equals EPLB off.
- Full option list and constraints: `examples/wide_ep/ep_load_balancer/README.md`, section "Adaptive EP Load Balancer".

### Disaggregated (PD) serving: decode only

Adaptive EPLB counts only steps whose batch holds generation requests alone, so it belongs on the generation
servers. Put the `load_balancer` block only in the `gen` worker config; leave the `ctx` workers without it:

```yaml
worker_config:
  ctx:
    moe_config:
      backend: MEGAMOE_CUTEDSL      # no load_balancer
  gen:
    enable_attention_dp: true
    moe_config:
      backend: MEGAMOE_CUTEDSL
      load_balancer:
        mode: adaptive
        num_slots: 512
        calibration_cache_dir: /shared/adaptive_eplb
```

On an aggregated server the context steps run on the installed placement without being counted.

## Known limits

- Prototype status (`status="prototype"` fields). Only `MEGAMOE_CUTEDSL` supports it; other MoE backends are
  rejected at model build. A MTP draft layer on another backend (e.g. TRTLLM-Gen via `moe_backend="AUTO"`) runs as
  plain EP outside the balancer.
- Single NVLink domain only; across domains it logs why and stays off. No MoE smart router.
- A runtime error (including ranks reaching different decisions) turns adaptive EPLB off for the rest of the run;
  serving continues on the installed placements.
- At most half of the replica slots hold replicas at any time (the other half receives the next install).
- `token_tile` return readiness on SM100/SM103: the kernel builds it for the NVLink-domain component only (not
  internode). The op offers it only to load-balanced autotuning with helper experts (per-layer dynamic EPLB), from
  2048 tokens per launch on; the default tactics and the adaptive-EPLB path keep expert readiness unless a tactic
  asks for it.
- Validated on GB300 (SM103), 4 GPUs per node, DeepSeek-V4-Pro NVFP4 GSM8K DEP4 with MTP; other models, EP sizes and
  multi-node MNNVL were not run on this exact branch tip.

## Validation

GB300 (SM103), image `1.4.0rc5.dev202611190007` with a C++ build of this branch (`3fb84c503` changes no C++ against
`2527f16ab`).

- GSM8K, DeepSeek-V4-Pro NVFP4, DEP4, MTP (3 drafts), adaptive EPLB on (`num_slots` = experts + 16): accuracy 96.51
  (reference 95.90), 7 installs, acceptance length 3.13.
- MegaMoE GPU unit tests against the parent: stage routes, staging, live rows, before-launch with MPI, proxy fences,
  adaptive route, MegaMoE e2e: no lost passes. token_tile on SM103 builds (`_fc2readytile_v1`) and equals expert
  readiness bit for bit with the fused shared expert, phase interleave on (4,1)/(2,1) clusters, the one-round router
  and the adaptive route; with helper slots it equals the helper-free launch.
- CPU: the commit's pre-commit hooks all pass; bandit reports one finding inherited from the parent
  (`adaptive_eplb/runtime.py:25`, B404 `import subprocess`).
- `test_megamoe_combine_dedup_tactic.py` (with the new regression test for the runner's kernel arguments) and
  `test_dynamic_eplb_contracts.py` pass on CPU in the image, run from a checkout that includes `cpp/`.

## Objective
- Implement "Shadow Copy" deployment in the HIP grafting codebase (`graftingmodular`): bake a `.graft` checkpoint into loaded base-model weights once at startup, free graft params, then run inference via native base-model GEMMs (no graft GEMMs, no backward, no AdamW). Then close the runtime proof (build, sentinel, training regression, explicit-vs-shadow, A/B/C reversibility, artifact hashing) on the ROCm host `root@100.119.181.4`.

## Important Details
- Host: `root@100.119.181.4` (tailscale IP 100.119.181.4), SSH already authenticated, ROCm 7.2.1, HIP 7.2.53211, gfx1100 (NOT MI300X). PyTorch at `/opt/venv/lib/python3.10/site-packages/torch`, ROCM at `/opt/rocm-7.2.1`.
- Phase 2 build command (from `/workspace/run_smoke.py`):
  `hipcc -std=c++17 -O3 -I/workspace -I/opt/venv/lib/python3.10/site-packages/torch/include -I/opt/venv/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -L/opt/venv/lib/python3.10/site-packages/torch/lib -L/opt/rocm-7.2.1/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lrocblas -lhipblaslt -Wl,-rpath,/opt/venv/lib/python3.10/site-packages/torch/lib -Wl,-rpath,/opt/rocm-7.2.1/lib --hip-link -o /workspace/grafting.shadow /workspace/grafting.hip`
- Model: `/workspace/model/real_SmolLM3-3B/model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors` (index json present). Eval/token bins: `/workspace/medical_domain.bin`, `/workspace/real_medical_domain.bin`, `/workspace/medical_ood.bin`, `/workspace/med_train.bin`, `/workspace/med_heldout.bin`.
- Shadow subsystem design: `apply_graft_to_shadow(ModelWeights&, const GraftWeights&)` bakes f32 deltas into bf16 base tensors in-place via one HIP kernel per projection; expand (gate/up) layout `base[i_start+s, h_start+h] += graft[s,h]`; contract (down) layout `base[h_start+h, i_start+s] += graft[s,h]` (transpose). Guard `g_shadow_baked` (inline bool in shadow_bake.hpp) prevents double-bake; `backward_layer` throws if `g_shadow_baked`.
- `forward_model`/`forward_layer` take `bool use_explicit_graft = true`; shadow passes `false`, training/eval default `true`.
- User requires sentinel with `h_slice_dim > blockDim.x` (S=3, H=513, kernel_threads=256) → kernel MUST loop `for (int h = threadIdx.x; h < h_slice_dim; h += blockDim.x)`. Current test22 does NOT do this (uses `int h = threadIdx.x;` only) — MUST FIX.
- User requires sentinel to exercise REAL `apply_graft_to_shadow()` projection-mask + layer-range logic, not a simulated skip.
- User questions whether `--eval-only` is a genuine explicit-eval path; verified: yes — it loads base + optional checkpoint, runs `forward_model(...,use_explicit_graft=true)`, computes CE/PPL, dumps `final_hidden.raw`/`logits.raw`, no backward/AdamW. (Confirmed in binary usage + code path at grafting.hip:561-667.)

## Work State
### Completed
- Created `src/shadow_bake.hpp` + `src/shadow_bake_impl.inc` (shadow bake kernel + `apply_graft_to_shadow`, `g_shadow_baked` inline guard).
- Modified `grafting.hip`: added `--shadow-eval` flag parsing, `bool shadow_eval`, shadow-eval runtime block (loads base, loads optional checkpoint, bakes, frees graft, runs forward with `use_explicit_graft=false`, prints proof counters + hashes), includes for shadow_bake files, `use_explicit_graft` param on `forward_model` + pass-through.
- Modified `src/layer_forward.hpp`/`.inc`: added `use_explicit_graft` param + `&& use_explicit_graft` guards on 3 graft GEMM gates.
- Modified `src/layer_backward_impl.inc`: added `g_shadow_baked` guard (throws if baked).
- Modified `src/types.hpp`: f32 cache comment for shadow mode.
- Modified `src/cli_impl.inc`: added `--shadow-eval`, `--eval-only`, `--eval-data` to usage.
- Fixed stale comment in `src/kernels_optimizer_impl.inc` (line 57-58).
- Created `tests/test22_shadow_bake_sentinel.cpp` (NEEDS kernel geometry fix — see Blocked).
- Created `tests/test23_training_regression_gate.py` (static guard only, not real training run).
- Updated OpenWiki docs.
- **[NEW] Fixed critical brace-imbalance bug in `grafting.hip`**: the training `else` block (opened at line 668) was closing prematurely at line 674 `}` (spurious), and a second spurious `if (eval_only)` at line 743 split the structure, leaving `domain`/`arena`/`graft` out of scope at the training loop and save section. Restructured so: validation `if (eval_only)` (538) now nests the eval-only inference block (561-667, returns); `else` (668) spans full training setup + training loop + save section; removed premature `}` at 674; moved eval-only block into the validation branch; fixed `else`-close placement after the save section. Verified `final_depth=0` and binary compiles + runs all three modes.

### Active
- Build is GREEN: `/workspace/grafting.shadow` (542KB, Jul 10 21:22) compiles clean, runs, shows all flags including `--shadow-eval` and `--eval-only`.
- Need to sync the fixed `grafting.hip` back to local `C:\graftingmodular` so the brace fix isn't lost.

### Blocked
- `tests/test22_shadow_bake_sentinel.cpp`: wrong kernel geometry (no `h += blockDim.x` loop) → must fix before compile/run. Must use REAL `apply_graft_to_shadow()` with projection masks + layer range (not simulated skip).
- Need actual runtime gates on host: (a) sentinel, (b) training regression, (c) explicit-vs-shadow, (d) A/B/C reversibility, (e) artifact hashing.

## Next Move
1. Sync fixed `grafting.hip` from host back to local repo.
2. Fix `tests/test22_shadow_bake_sentinel.cpp`: change kernel to loop `for (int h = threadIdx.x; h < h_slice_dim; h += blockDim.x)`, set H=513, kernel_threads=256; call REAL `apply_graft_to_shadow()` with projection masks + layer range.
3. Compile + run sentinel on host; capture stdout + exit code.
4. Run `--shadow-eval` and `--eval-only` on `/workspace/model/real_SmolLM3-3B` with `/workspace/real_medical_domain.bin`; compare logits_sum/logits_hash.
5. Run a short training regression (small `--steps`) to confirm `domain`/`ood`/`arena`/`graft` in scope and loop executes.
6. A/B/C reversibility + artifact hashing.

## Relevant Files
- `/workspace/grafting.hip` (host) / `C:\graftingmodular\grafting.hip` (local): main single-TU source; shadow_eval block + use_explicit_graft; **brace bug fixed**.
- `/workspace/grafting.hip.bak` + `.bak2`: originals for diff/reference.
- `C:\graftingmodular\src\shadow_bake.hpp` + `src\shadow_bake_impl.inc`: shadow bake subsystem.
- `C:\graftingmodular\src\layer_forward.hpp` / `.inc`: use_explicit_graft param + guards.
- `C:\graftingmodular\src\layer_backward_impl.inc`: g_shadow_baked guard.
- `C:\graftingmodular\src\cli_impl.inc`: usage text.
- `C:\graftingmodular\src\types.hpp`: f32 cache comment.
- `C:\graftingmodular\tests\test22_shadow_bake_sentinel.cpp`: NEEDS kernel geometry fix.
- `C:\graftingmodular\tests\test23_training_regression_gate.py`: static guard only.
- `/workspace/model/real_SmolLM3-3B/`: base model safetensors shards.
- `/workspace/real_medical_domain.bin`, `/workspace/medical_domain.bin`: eval data.

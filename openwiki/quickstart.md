# Grafting (Axis ARW) — Quickstart

**Grafting** (also called Axis ARW — Axis-Aligned Residual Weights) is a multi-domain fine-tuning technique for frozen Transformer language models. It trains small delta-weight matrices ("grafts") that slide into hard axis-aligned slices of the base model's FFN weights, allowing multiple domain experts to coexist in a single model without catastrophic interference.

The project targets **SmolLM3-3B** on **AMD MI300X GPUs** and comprises two parallel codebases:

- **HIP binary** (`grafting.hip` + 35 files in `src/`): High-performance training engine using ROCm HIP kernels, rocBLAS/hipBLASLt GEMM, and custom fused elementwise kernels. Compiles as a single translation unit.
- **Python TPHS** (`grafting/`): Pure PyTorch training, evaluation, stacking, and install pipeline with the same axis-aligned grafting math.

## Repository Structure

| Path | Purpose |
|------|---------|
| `grafting.hip` | Orchestrator — `main()`, CLI, training loop, checkpoint I/O |
| `src/*.hpp` + `src/*.inc` | 19 header + 16 implementation files (single TU pattern) |
| `grafting/` | Python package (TPHS): train, eval, stack, install, dataset, autopsy |
| `tests/` | 22+ test files: contract operations, precision budget, sentinel tests |
| `docs/` | Research documentation: contract audit, file ownership, stack accounting, mechanism reports |
| `diagnostics/` | Empirical telemetry data (JSON/JSONL) from MI300X training runs |
| `.github/workflows/openwiki-update.yml` | CI workflow for automated documentation refresh |

## Quick Links

- [Architecture Overview](architecture/overview.md) — System design, single-TU compile model, two-codebase architecture
- [Graft Mechanism](architecture/graft-mechanism.md) — Core technique: axis-aligned slicing, delta weights, silence blend, expansion trap
- [Stack Accounting Model](domain/stack-accounting-model.md) — How multiple grafts coexist, additivity findings, OOD behavior, precision budget
- [Training Pipeline](operations/training-pipeline.md) — How to train, evaluate, stack, and bake grafts
- [Verification & Testing](testing/verification.md) — Contract audit, kernel harnesses, sentinel tests, parity status

## Key Findings (at 200 steps, SmolLM3-3B)

| Domain | Base PPL | Grafted PPL | Delta |
|--------|----------|-------------|-------|
| Finance | 92.80 | 1.44 | **-91.35** |
| Medical | 15.65 | 1.71 | **-13.94** |
| Legal | 8.15 | 3.81 | **-4.34** |
| Coding | 2.87 | 1.28 | **-1.59** |

The graft surgically targets the base model's ignorance — correlation between base-model perplexity and graft-induced loss reduction is **+0.97**.

## Core Principle

Each graft artifact (~300 MB) contains delta weights for gate, up, and down projections at every FFN layer. These deltas occupy disjoint index ranges (axis-aligned slices) of the weight matrices, so multiple grafts can be stacked into the same model without the cross-domain interference that plagues rotated-subspace approaches.

> **Version lineage**: v0.1 used rotated orthogonal subspaces (broken by SwiGLU nonlinearity). v0.2 used energy-matched injection (optimizer fought back, causing degenerate logit explosions). v0.3 (current) uses hard axis-aligned slicing.

## Reference

- Base model: `HuggingFaceTB/SmolLM3-3B`
- Hardware: AMD MI300X (130 GB VRAM recommended for 4-domain stack)
- Framework: ROCm 7.2+, PyTorch 2.4+, HIP/ROCm for custom kernels

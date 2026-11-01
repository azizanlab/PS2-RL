# PS2-RL checkpoint bundle

This directory holds the pretrained artifacts needed to reproduce the paper's
Phase-2 runs and the reset-library / warmstart pipelines, plus representative
**deployed Phase-1 SA policies and Phase-2 PS2 policies** you can evaluate/deploy without a 5–14 h retrain.
git-LFS / external hosting is unnecessary), so the commands below resolve out of the box
after cloning.

## Contents

Three groups: `deployed_sa/` (Phase-1 safe-arrival policies), `deployed_ps2/`
(Phase-2 PS2 policies), and `quadrotor_vanilla/` (the vanilla-tracker warm-start +
its derived reset library + source traces).

| Path | What it is | Consumed by |
|---|---|---|
| `deployed_sa/unicycle_sa/best_weights.pkl` | Unicycle Phase-1 safe-arrival backup actor | `train_phase2 --system unicycle` (learned backup), `evaluate_phase1` |
| `deployed_sa/quadrotor_sa/best_weights.pkl` | Quadrotor Phase-1 safe-arrival backup actor | `train_phase2 --system quadrotor` (learned backup), `evaluate_phase1` |
| `deployed_sa/quadrotor_sa/configs.json` | Run config for the quad SA actor (`actor` net arch + `backup_env`/`reset_library`; CBF params at `backup_env.cbf_cfg`) | loaded alongside `deployed_sa/quadrotor_sa/best_weights.pkl` |
| `quadrotor_vanilla/quadrotor_vanilla_weights.pkl` | Vanilla-tracker SAC weights used to warm-start quad Phase-2 | `train_phase2 --system quadrotor` |
| `quadrotor_vanilla/reset_library.pkl` | Quad Phase-1 reset library: the pools/splits/nominal/held-out reset **states** plus the minimal base-set/safe-set config needed to sample them | `train_phase1 --system quadrotor` (reference; rebuilt from the traces) |
| `quadrotor_vanilla/vanilla_traces/trace_seed0_runIdx{NN}.npz` | 20 vanilla-eval episode traces (`NN` ∈ 00–19) | reset-library construction |
| `deployed_ps2/<family>/{best_weights.pkl, configs.json}` | Deployed Phase-2 PS2 policies (4 families) — a trained safe controller you can evaluate/deploy directly | `evaluate_phase2 --experiment checkpoints/deployed_ps2/<family>` |

The *Consumed by* column names the release entrypoints
(`train_phase1` / `train_phase2` / `evaluate_phase1` / `evaluate_phase2` with `--system …`).

The **powerloop reference trajectory** is not stored here — it ships in the package at
`ps2rl/envs/assets/quadrotor_powerloop_reference.npz` (byte-identical to the research
source).

### Reset-library trace selection

The Phase-1 quadrotor reset library is built from the **20** shipped traces
(`seed_0`, `runIdx 00–19`), in numeric `runIdx` order. The `trace_seed0_runIdx{NN}.npz`
naming makes this selection explicit and reproducible; `train_phase1 --system quadrotor`
rebuilds the library from them and reproduces the shipped `reset_library.pkl`.

### Deployed Phase-2 PS2 policies

`deployed_ps2/` ships one representative **deployed** PS2 policy per config family, so
you can evaluate/deploy the safe controller without retraining Phase 2. Each family is a
minimal run directory (`configs.json` + the checkpoint), so the release evaluator loads
it directly:

| Family (`deployed_ps2/…`) | Backup policy | Training seed | Selected training step |
|---|---|---:|---:|
| `unicycle_ps2_analytic` | analytic (LQR) | 9 | 610,000 |
| `unicycle_ps2_learned` | learned (Phase-1 SA) | 2 | 660,000 |
| `quadrotor_ps2_analytic` | analytic (cascaded PID + LQR) | 4 | 1,500,000 |
| `quadrotor_ps2_learned` | learned (Phase-1 SA) | 9 | 1,495,000 |

Every family's checkpoint is `best_weights.pkl` — the checkpoint the release
Phase-2 evaluator selects (`--weight_preference best_only`). All four policies were
trained with the hard QP. Within each source run, the saved checkpoint was selected
by maximum training-evaluation return. For quadrotor, the source filename was
`best_weights_return.pkl`; staging changes only its filename. The unicycle run
choices have the lowest lateral position MAE among the evaluated checkpoints in
their respective backup families. The chosen quadrotor runs use the return-selected
checkpoint rather than the separate position-selected checkpoint.

The `configs.json` files use the release schema. The source terminal gain maps to
`base_alpha`, and the equal terminal/capture levels map to `base_set_c` (`0.3` for
unicycle and `8.0` for quadrotor). Quadrotor LQR Q/R entries are the source cost rates
multiplied by `dt=0.02`; the release consumes those discrete-time costs directly.
These conversions preserve the base controller and base set. Backup and warm-start
paths resolve to the staged artifacts, and the empty quadrotor reference path selects
the packaged powerloop reference. The existing Phase-1 actor arrays are unchanged.

The staged Phase-2 files are byte-identical to the selected source weight files:

| Family | Source run timestamp | SHA-256 of `best_weights.pkl` |
|---|---|---|
| `unicycle_ps2_analytic` | `20261027_201014` | `f0064a3642fad35534622ab9d17d475fca615a4ea2365e6650af71f6d17a5337` |
| `unicycle_ps2_learned` | `20260724_122253` | `11e5453e04992bff232a9685345fc52625f5a40fdf1db552bc764e8603a47ab9` |
| `quadrotor_ps2_analytic` | `20261028_002933` | `2e55da4f5ad28a07839c3eb42ec5d2621a140014b8bde378bb76d4b6557fbfb8` |
| `quadrotor_ps2_learned` | `20261028_134127` | `79a53ac779a7dd57ea91cd7c0917a05d2042d5cdd3c1278b8d13b27c3dff1f19` |

Run the evaluator from the repo root so those relative paths resolve, e.g.:

```bash
python scripts/evaluate_phase2.py --system quadrotor \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/quadrotor_ps2_learned \
  --weight_preference best_only

python scripts/evaluate_phase2.py --system unicycle \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/unicycle_ps2_learned \
  --weight_preference best_only
```


<h1 align="center">PS2-RL</h1>

<p align="center">
  <b><a href="https://arxiv.org/abs/2606.14536">Provably Safe, yet Scalable Reinforcement Learning</a></b>
  <br><br>
  <a href="https://kaiyun717.github.io/">Kai S. Yun</a> &nbsp;·&nbsp; <a href="https://scholar.google.com/citations?user=gksmrPsAAAAJ&hl=en">Zeyang Li</a> &nbsp;·&nbsp; <a href="https://azizan.mit.edu/index.html">Navid Azizan</a>
  <br>
  <i>Massachusetts Institute of Technology</i>
  <br><br>
  <b>NeurIPS <span style="color: darkred;">Spotlight</span> 2026, Sydney</b>
</p>


PS2-RL trains reinforcement-learning policies that are **safe by construction** without sacrificing scalability. Every action a policy proposes is passed through a differentiable **Control-Invariant Layer (CIL)** that projects it onto the set of actions guaranteed to keep the system inside a forward-invariant safe set. Because the layer is differentiable, the policy is trained *end-to-end through it* with any standard RL backbone, so it learns to perform well while safety is enforced at every step, at train and deploy time alike.

The whole stack is **JAX-native and GPU-accelerated** (`jit` / `lax.scan` / `vmap`).

---

## Method at a glance

Two testbeds, each with a two-phase pipeline:

| Testbed | State | Task |
|---|---|---|
| **Unicycle** | 3D | lane-keeping / trajectory tracking |
| **Quadrotor** | 10D | aggressive power-loop tracking |

- **Phase 1 — safe-arrival policy** is trained so that, composed with an LQR **base controller** on the base set *B*, it forms a **learned backup policy** that drives the system into the base set while staying safe. This is the certificate the CIL leans on. *(You can also use a purely **analytic** backup — no Phase-1 training required.)*
- **Phase 2 — PS2 policy** is trained end-to-end **through the CIL**: every actor/critic action is projected by the differentiable BCBF-QP ([HardNet-CVX](https://arxiv.org/abs/2410.10807) + [qpax](https://github.com/kevin-tracy/qpax)) against the backup policy from Phase 1 (or the analytic backup).

## Repository layout

```
ps2rl/                     # the library (JAX)
  envs/                    #   unicycle + quadrotor dynamics/environments (+ powerloop reference asset)
  base_controller/         #   discrete-time LQR base controllers
  sets/                    #   safe sets and base sets
  backup_policy/           #   analytic + learned backup policies
  cil/                     #   the Control-Invariant Layer
  phase1_sa/               #   Phase-1 safe-arrival trainers
  phase2_ps2/              #   Phase-2 PS2 trainers
  evaluation/              #   evaluation pipelines
  plotting/  utils/        #   plotting + shared utilities
scripts/                   # command-line entrypoints (see Quickstart)
slurm_batch/               # Slurm drivers that reproduce the paper runs
checkpoints/               # pretrained policies + reset library (see checkpoints/README.md)
requirements.txt           # pinned dependencies
```

## Installation

Tested with Python **3.10** on Ubuntu 24.04.4 LTS. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `jax[cuda12]==0.6.2` / `jaxlib==0.6.2` (CUDA 12), `numpy`, `scipy`, `matplotlib`, and `qpax==0.0.9` (the differentiable QP solver).

- **GPU (recommended for training/eval):** `jax[cuda12]` pulls the matching CUDA 12 JAX plugins for a standard install on a CUDA-12 machine. Training and evaluation should always run on GPU — the environments and QP are far too slow on CPU.
- **CPU (only for `--help` / import smoke tests):** `pip install jax==0.6.2` suffices, and
  set `JAX_PLATFORMS=cpu`.

Smoke-test the install:

```bash
python -c "import ps2rl, qpax, jax; print('ok', jax.__version__)"
JAX_PLATFORMS=cpu python scripts/evaluate_phase2.py --system unicycle --help
```

## Quickstart

All commands are run from the repository root and dispatch on `--system {unicycle,quadrotor}`.

### 1. Evaluate a deployed policy (no training required)

The `checkpoints/` bundle ships four ready-to-run **deployed PS2 policies** plus the Phase-1 safe-arrival policies and the quadrotor warm-start/reset library, so you can evaluate a trained *safe* controller immediately:

```bash
# Quadrotor PS2 policy with a learned Phase-1 backup
python scripts/evaluate_phase2.py --system quadrotor \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/quadrotor_ps2_learned \
  --weight_preference best_only

# Unicycle PS2 policy with a learned Phase-1 backup
python scripts/evaluate_phase2.py --system unicycle \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/unicycle_ps2_learned \
  --weight_preference best_only
```

The four families are `{unicycle,quadrotor}_ps2_{analytic,learned}`. See
[`checkpoints/README.md`](checkpoints/README.md) for the full bundle contents.

### 2. Train

```bash
# Phase 1 — safe-arrival backup policy
python scripts/train_phase1.py --system unicycle    [options]
python scripts/train_phase1.py --system quadrotor   [options]

# Phase 2 — PS2 policy (SAC through the Control-Invariant Layer)
python scripts/train_phase2.py --system unicycle    [options]
python scripts/train_phase2.py --system quadrotor   [options]
```

Phase 2 uses a Phase-1 backup policy: pass `--backup_policy_mode learned` with a Phase-1 checkpoint (e.g. `checkpoints/deployed_sa/…/best_weights.pkl`), or `--backup_policy_mode analytic` for the analytic backup (no Phase-1 training needed). Run any entrypoint with `--system <sys> --help` for the full option list.

### 3. Evaluate

```bash
# Phase 1: held-out scoring of a learned backup policy, or learned-vs-analytic comparison
python scripts/evaluate_phase1.py --system quadrotor --mode score   [options]
python scripts/evaluate_phase1.py --system quadrotor --mode compare [options]

# Phase 2: evaluate a trained PS2 policy directory
python scripts/evaluate_phase2.py --system unicycle [options]
```

Two convenience entrypoints exist as thin wrappers: `scripts/train_vanilla_tracker.py` (≡ `train_phase2.py --system quadrotor` with the projection disabled — the vanilla tracker used to warm-start the quadrotor) and `scripts/compare_backup_policies.py` (≡ `evaluate_phase1.py --system quadrotor --mode compare`).

## Reproducing the paper

The `slurm_batch/` directory holds the Slurm drivers that reproduce the reported runs (within a tolerance), and each pins the canonical hyperparameters and writes results under `outputs/`.

| Driver | Reproduces |
|---|---|
| `slurm_train_uni_phase1.sh` / `slurm_train_quad_phase1.sh` | Phase-1 safe-arrival backup policies |
| `slurm_train_uni_phase2.sh` / `slurm_train_quad_phase2.sh` | Phase-2 PS2 policies (learned + analytic backup, 10 seeds) |
| `slurm_train_quad_vanilla.sh` | quadrotor vanilla tracker (warm-start source) |
| `slurm_eval_{uni,quad}_phase{1,2}.sh` | the corresponding evaluations |

Submit from the repository root, e.g. `sbatch slurm_batch/slurm_train_uni_phase2.sh`. The drivers target a Supercloud-style cluster (`module load anaconda/Python-ML-2025a`, partition `xeon-g6-volta`, V100 GPU, `JAX_PLATFORMS=cuda`); adapt the `#SBATCH` headers to your environment. Approximate single-GPU training times from the paper: Phase-1 safe-arrival ≈ **840 s** (unicycle) / **2400 s** (quadrotor); Phase-2 PS2 ≈ **4.6 h** (unicycle) / **13.6 h** (quadrotor).

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{ps2rl,
      title={Provably Safe, Yet Scalable Reinforcement Learning}, 
      author={Kai S. Yun and Zeyang Li and Navid Azizan},
      year={2026},
      eprint={2606.14536},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.14536}, 
}
```

## License

Released under the [MIT License](LICENSE) — © 2026 Azizan Lab, Massachusetts Institute of
Technology.

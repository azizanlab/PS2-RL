#!/bin/bash
# Quadrotor Phase-2 (PS2) policy evaluation over a directory of trained runs.
# Each array task evaluates one run directory and writes results into its
# evaluation/ subdir. Entrypoint: scripts/evaluate_phase2.py --system quadrotor.

#SBATCH --job-name=quad-p2-eval
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=16:00:00
#SBATCH --array=0-19
#SBATCH --qos=high
#SBATCH -o slurm-quad-p2-eval-%A_%a.out
#SBATCH -e slurm-quad-p2-eval-%A_%a.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

# In Slurm batch mode, BASH_SOURCE may point to a scheduler copy; use submit dir.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
    echo "Could not resolve PROJECT_ROOT from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}; got ${PROJECT_ROOT}"
    exit 1
fi

# ----------------------------- User settings -----------------------------

outputs_dir_name="outputs/quad_phase2_ps2"
run_glob="*"

num_eval_seeds=20
episodes_per_seed=50
eval_seed_base=3500000
seed_group_stride=10000
label="quad_eval1000"
weight_preference="best_only"    # both | best_only | final_only  (best_only = the return-selected best_weights.pkl)
parallel_workers=1
rollout_batch_size=128
gif_slowdown="0.25"
gif_trail_length="150"
gif_print_every="25"

# -----------------------------------------------------------------------

if [[ "${outputs_dir_name}" = /* ]]; then
    outputs_root="${outputs_dir_name}"
else
    outputs_root="${PROJECT_ROOT}/${outputs_dir_name}"
fi

if [ ! -d "${outputs_root}" ]; then
    echo "Outputs root does not exist: ${outputs_root}"
    exit 1
fi

mapfile -t experiments < <(
    find "${outputs_root}" -mindepth 1 -maxdepth 1 -type d -name "${run_glob}" -printf '%f\n' | sort
)

num_experiments="${#experiments[@]}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${num_experiments}" -eq 0 ]; then
    echo "No experiments found under ${outputs_root} matching '${run_glob}'"
    exit 1
fi

if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${num_experiments}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((num_experiments - 1))]"
    echo "Discovered ${num_experiments} experiments under ${outputs_root}"
    exit 1
fi

exp_name="${experiments[$task_id]}"
run_dir="${outputs_root}/${exp_name}"
if [ ! -d "${run_dir}" ]; then
    echo "Run directory does not exist: ${run_dir}"
    exit 1
fi
if [ ! -f "${run_dir}/configs.json" ] && [ ! -f "${run_dir}/config.json" ]; then
    echo "Missing configs.json/config.json under run directory: ${run_dir}"
    exit 1
fi

eval_dir="${run_dir}/evaluation"
mkdir -p "${eval_dir}"
tag="$(date +%Y%m%d_%H%M%S)"
tmp_log="${eval_dir}/quad_p2_eval-${tag}-seeds_${num_eval_seeds}-epPerSeed_${episodes_per_seed}-${label}.log"
exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a
cd "${PROJECT_ROOT}"

export JAX_PLATFORMS="cuda"
export JAX_PLATFORM_NAME="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-quad-p2-eval-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-quad-p2-eval-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/evaluate_phase2.py --system quadrotor
    --outputs_dir "${outputs_dir_name}"
    --experiment "${exp_name}"
    --num_eval_seeds "${num_eval_seeds}"
    --episodes_per_seed "${episodes_per_seed}"
    --eval_seed_base "${eval_seed_base}"
    --seed_group_stride "${seed_group_stride}"
    --weight_preference "${weight_preference}"
    --parallel_workers "${parallel_workers}"
    --rollout_batch_size "${rollout_batch_size}"
    --eval_label "${label}"
    --gif_slowdown "${gif_slowdown}"
    --gif_trail_length "${gif_trail_length}"
    --gif_print_every "${gif_print_every}"
)

echo "[$(date)] Starting quad Phase-2 (PS2) saved-policy eval"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}"
echo "outputs_dir=${outputs_dir_name}"
echo "run_glob=${run_glob}"
echo "num_experiments=${num_experiments}"
echo "experiment=${exp_name}"
echo "run_dir=${run_dir}"
echo "num_eval_seeds=${num_eval_seeds}"
echo "episodes_per_seed=${episodes_per_seed}"
echo "episodes_total_per_checkpoint=$((num_eval_seeds * episodes_per_seed))"
echo "weight_preference=${weight_preference}"
echo "parallel_workers=${parallel_workers}"
echo "rollout_batch_size=${rollout_batch_size}"
echo "log_file=${tmp_log}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
python - <<'PY'
import jax
print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())
PY
echo
"${cmd[@]}"
echo "[$(date)] Quad Phase-2 eval complete"

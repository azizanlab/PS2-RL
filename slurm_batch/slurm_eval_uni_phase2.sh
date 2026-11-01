#!/bin/bash
# Unicycle Phase-2 (PS2) lane-keeping policy evaluation over a directory of
# trained runs. Each array task evaluates one run directory and writes results
# into its evaluation/ subdir. Entrypoint: scripts/evaluate_phase2.py --system unicycle.
#
# NOTE: output/run directory names retain the internal reproduction codename
# (outputs_objD_*) so paths match the ground-truth artifacts in backup-RL.

#SBATCH --job-name=uni-p2-eval
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --qos=high
#SBATCH -o slurm-uni-p2-eval-%A_%a.out
#SBATCH -e slurm-uni-p2-eval-%A_%a.err

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

outputs_dir_name="outputs/uni_phase2_ps2"
run_glob="*"

episodes=1000
max_steps=400
env_dt=0.05
eval_seed_base=2500000
label="traj_eval1000"
weight_preference="best_only"   # both | best_only | final_only

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
    exit 1
fi

exp_name="${experiments[$task_id]}"
run_dir="${outputs_root}/${exp_name}"
eval_dir="${run_dir}/evaluation"
mkdir -p "${eval_dir}"
tag="$(date +%Y%m%d_%H%M%S)"
tmp_log="${eval_dir}/trajSafe_eval-${tag}-ep_${episodes}-ms_${max_steps}-${label}.log"
exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a
cd "${PROJECT_ROOT}"

export JAX_PLATFORMS="cuda"
export JAX_PLATFORM_NAME="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-uni-p2-eval-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-uni-p2-eval-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/evaluate_phase2.py --system unicycle
    --outputs_dir "${outputs_dir_name}"
    --experiment "${exp_name}"
    --episodes "${episodes}"
    --max_steps "${max_steps}"
    --env_dt "${env_dt}"
    --eval_seed_base "${eval_seed_base}"
    --weight_preference "${weight_preference}"
    --parallel_workers 1
    --eval_label "${label}"
    --eval_v_ref_mode saved
)

echo "[$(date)] Starting uni Phase-2 (PS2) traj-track eval run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}"
echo "outputs_dir=${outputs_dir_name}"
echo "run_glob=${run_glob}"
echo "num_experiments=${num_experiments}"
echo "experiment=${exp_name}"
echo "run_dir=${run_dir}"
echo "log_file=${tmp_log}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"
echo "[$(date)] Uni Phase-2 eval complete"

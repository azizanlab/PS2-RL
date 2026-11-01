#!/bin/bash
# Unicycle Phase-1: rerun the learned-vs-analytic invariant-set comparison on a
# trained backup-policy checkpoint at the fine paper grid (201x201x121), writing
# results into a subdir under the run. Entrypoint: scripts/evaluate_phase1.py --system unicycle.
#
# Each array task processes one trained Phase-1 run under the train output root.

#SBATCH --job-name=uni-p1-eval
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --mem=64G
#SBATCH --qos=high
#SBATCH -o slurm-uni-p1-eval-%A_%a.out
#SBATCH -e slurm-uni-p1-eval-%A_%a.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
    echo "Could not resolve PROJECT_ROOT from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}; got ${PROJECT_ROOT}"
    exit 1
fi

run_root_rel="outputs/uni_phase1_sapolicy"
run_root="${PROJECT_ROOT}/${run_root_rel}"
run_glob="*"
checkpoint_name_val="best_weights.pkl"
output_subdir_val="invariant_compare_best_analytic_withLQRtail_pidSameAsLQR_saved"

compare_num_y_val="201"
compare_num_psi_val="201"
compare_num_v_val="121"
compare_v_min_val="0.0"
compare_v_max_val="12.0"
compare_max_scatter_points_val="20000"

# Analytic-baseline gains (echoed to metadata; the analytic backup policy uses
# the LQR gain matrix K directly, so these are informational).
analytic_kv_val="7.8078"
analytic_ky_y_val="1.2790"
analytic_ky_psi_val="3.9614"

if [ ! -d "${run_root}" ]; then
    echo "Run root not found: ${run_root}"
    exit 1
fi

mapfile -t run_dirs < <(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -name "${run_glob}" | sort)
num_runs="${#run_dirs[@]}"
if [ "${num_runs}" -le 0 ]; then
    echo "No Phase-1 run directories found under ${run_root} matching '${run_glob}'"
    exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${num_runs}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((num_runs - 1))]"
    exit 1
fi

run_dir="${run_dirs[$task_id]}"

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-uni-p1-eval-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-uni-p1-eval-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/evaluate_phase1.py --system unicycle
    --run_dir "${run_dir}"
    --checkpoint_name "${checkpoint_name_val}"
    --output_subdir "${output_subdir_val}"
    --compare_num_y "${compare_num_y_val}"
    --compare_num_psi "${compare_num_psi_val}"
    --compare_num_v "${compare_num_v_val}"
    --compare_v_min "${compare_v_min_val}"
    --compare_v_max "${compare_v_max_val}"
    --compare_max_scatter_points "${compare_max_scatter_points_val}"
    --analytic_kv "${analytic_kv_val}"
    --analytic_ky_y "${analytic_ky_y_val}"
    --analytic_ky_psi "${analytic_ky_psi_val}"
    --save-results
)

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}/${num_runs}"
echo "run_root=${run_root}"
echo "run_dir=${run_dir}"
echo "checkpoint_name=${checkpoint_name_val}"
echo "output_subdir=${output_subdir_val}"
echo "analytic_kv=${analytic_kv_val} analytic_ky_y=${analytic_ky_y_val} analytic_ky_psi=${analytic_ky_psi_val}"
printf 'cmd='
printf ' %q' "${cmd[@]}"
printf '\n'

nvidia-smi || true
echo
"${cmd[@]}"
echo "[$(date)] Invariant-compare rerun complete"

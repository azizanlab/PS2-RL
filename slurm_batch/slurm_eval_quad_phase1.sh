#!/bin/bash
# Quadrotor Phase-1 learned-vs-analytic backup-policy comparison near the
# powerloop (learned safe-arrival backup policy vs the analytic hybrid PID+dLQR
# backup). Each array task compares one trained Phase-1 run; results are written
# under that run's comparison/ folder. Entrypoint:
# scripts/evaluate_phase1.py --system quadrotor --mode compare
# (equivalently scripts/compare_backup_policies.py).

#SBATCH --job-name=quad-p1-eval
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=1:00:00
#SBATCH --qos=high
#SBATCH --array=0-4
#SBATCH -o slurm-quad-p1-eval-%A_%a.out
#SBATCH -e slurm-quad-p1-eval-%A_%a.err

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

RUN_ROOT_REL="outputs/quad_phase1_sapolicy"
RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT_REL}"
if [ ! -d "${RUN_ROOT}" ]; then
    echo "Missing RA run root: ${RUN_ROOT}"
    exit 1
fi

mapfile -t RUN_DIRS < <(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '*' | sort)
num_lbps="${#RUN_DIRS[@]}"
if [ "${num_lbps}" -le 0 ]; then
    echo "No learned RA run directories found under ${RUN_ROOT}"
    exit 1
fi

benchmark_seed_val="0"
num_perturbed_general_val="1024"
num_perturbed_near_ceiling_val="1024"
num_perturbed_bridge_val="1024"
num_perturbed_capture_shell_val="512"
batch_size_val="512"
compare_policies_raw="${COMPARE_POLICIES:-}"
save_rollout_trajectories_raw="${COMPARE_SAVE_ROLLOUT_TRAJECTORIES:-0}"

compare_policies_tag=""
if [ -n "${compare_policies_raw}" ]; then
    compare_policies_tag="$(echo "${compare_policies_raw//,/ }" | tr -s ' ' '-' | sed 's/^-//; s/-$//')"
fi

save_rollout_trajectories_flag="1"
case "${save_rollout_trajectories_raw}" in
    1|true|TRUE|yes|YES|on|ON)
        save_rollout_trajectories_flag="1"
        ;;
esac

# Analytic hybrid PID gains, fixed to the reported (App. F.4) settings.
pid_kp_z_val="36.0"
pid_kv_z_val="24.0"
pid_kv_xy_val="14.0"
pid_attitude_p_gain_val="45.0"
pid_z_safety_gain_val="32.0"
pid_ceiling_vz_gain_val="18.0"
pid_yaw_gain_scale_val="0.35"
pid_ceiling_margin_val="0.60"
pid_lateral_boost_val="1.50"
pid_min_virtual_accel_z_val="0.0"

total_runs="${num_lbps}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total_runs}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((total_runs - 1))]"
    exit 1
fi

RUN_DIR="${RUN_DIRS[$task_id]}"
RUN_DIR_REL="${RUN_DIR#${PROJECT_ROOT}/}"
if [ ! -f "${RUN_DIR}/configs.json" ]; then
    echo "Missing configs.json for learned policy run: ${RUN_DIR}"
    exit 1
fi

tagify() {
    local raw="$1"
    raw="${raw//-/_m_}"
    raw="${raw//./p}"
    echo "${raw}"
}

kpz_tag="$(tagify "${pid_kp_z_val}")"
kvz_tag="$(tagify "${pid_kv_z_val}")"
kvxy_tag="$(tagify "${pid_kv_xy_val}")"
attp_tag="$(tagify "${pid_attitude_p_gain_val}")"
yaw_tag="$(tagify "${pid_yaw_gain_scale_val}")"
cm_tag="$(tagify "${pid_ceiling_margin_val}")"
zsafe_tag="$(tagify "${pid_z_safety_gain_val}")"
cvz_tag="$(tagify "${pid_ceiling_vz_gain_val}")"
lat_tag="$(tagify "${pid_lateral_boost_val}")"
minz_tag="$(tagify "${pid_min_virtual_accel_z_val}")"

suffix="allRA-id${task_id}-kpz${kpz_tag}-kvz${kvz_tag}-kvxy${kvxy_tag}-attp${attp_tag}-yaw${yaw_tag}-cm${cm_tag}-zsafe${zsafe_tag}-cvz${cvz_tag}-lat${lat_tag}-minz${minz_tag}"
if [ -n "${compare_policies_tag}" ]; then
    suffix="${suffix}-pol${compare_policies_tag}"
fi
if [ "${save_rollout_trajectories_flag}" = "1" ]; then
    suffix="${suffix}-traj"
fi
comparison_root="${RUN_DIR}/comparison"
tmp_log="${comparison_root}/compare-${suffix}.train.log"
mkdir -p "${comparison_root}"
exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-quad-p1-eval-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-quad-p1-eval-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/evaluate_phase1.py --system quadrotor --mode compare
    --run_dir "${RUN_DIR_REL}"
    --seed "${benchmark_seed_val}"
    --num_perturbed_general "${num_perturbed_general_val}"
    --num_perturbed_near_ceiling "${num_perturbed_near_ceiling_val}"
    --num_perturbed_bridge "${num_perturbed_bridge_val}"
    --num_perturbed_capture_shell "${num_perturbed_capture_shell_val}"
    --batch_size "${batch_size_val}"
    --pid_kp_z "${pid_kp_z_val}"
    --pid_kv_z "${pid_kv_z_val}"
    --pid_kv_xy "${pid_kv_xy_val}"
    --pid_attitude_p_gain "${pid_attitude_p_gain_val}"
    --pid_yaw_gain_scale "${pid_yaw_gain_scale_val}"
    --pid_ceiling_margin "${pid_ceiling_margin_val}"
    --pid_z_safety_gain "${pid_z_safety_gain_val}"
    --pid_ceiling_vz_gain "${pid_ceiling_vz_gain_val}"
    --pid_lateral_boost "${pid_lateral_boost_val}"
    --pid_min_virtual_accel_z "${pid_min_virtual_accel_z_val}"
    --output_suffix "${suffix}"
)

if [ -n "${compare_policies_raw}" ]; then
    read -r -a compare_policies <<< "${compare_policies_raw//,/ }"
    if [ "${#compare_policies[@]}" -gt 0 ]; then
        cmd+=(--policies)
        for policy_name in "${compare_policies[@]}"; do
            if [ -n "${policy_name}" ]; then
                cmd+=("${policy_name}")
            fi
        done
    fi
fi

if [ "${save_rollout_trajectories_flag}" = "1" ]; then
    cmd+=(--save-rollout-trajectories)
fi

echo "[$(date)] Starting quad Phase-1 backup-policy compare run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}/${total_runs}"
echo "num_lbps=${num_lbps}"
echo "run_dir=${RUN_DIR}"
echo "benchmark_seed=${benchmark_seed_val} batch_size=${batch_size_val}"
echo "perturbed_general=${num_perturbed_general_val} perturbed_near_ceiling=${num_perturbed_near_ceiling_val} perturbed_bridge=${num_perturbed_bridge_val} perturbed_capture_shell=${num_perturbed_capture_shell_val}"
echo "ABP PID: kp_z=${pid_kp_z_val} kv_z=${pid_kv_z_val} kv_xy=${pid_kv_xy_val} attitude_p=${pid_attitude_p_gain_val} yaw_scale=${pid_yaw_gain_scale_val} ceiling_margin=${pid_ceiling_margin_val} z_safety=${pid_z_safety_gain_val} ceiling_vz=${pid_ceiling_vz_gain_val} lateral_boost=${pid_lateral_boost_val} min_virtual_accel_z=${pid_min_virtual_accel_z_val}"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"

latest_dir="$(ls -td "${comparison_root}"/quadBackup_compare-*-"${suffix}" 2>/dev/null | head -n 1 || true)"
if [ -n "${latest_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${latest_dir}/compare.log"
fi
echo "[$(date)] Compare run complete"

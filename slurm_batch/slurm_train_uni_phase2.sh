#!/bin/bash
# Unicycle Phase-2 PS2 policy training (SAC through the control-invariant layer),
# comparing the learned backup policy (PS2_SA) against the analytic LQR backup
# (PS2_ABP). 10 seeds x 2 backup modes = 20 runs. Entrypoint:
# scripts/train_phase2.py --system unicycle.
#
# NOTE: output root retains the internal reproduction codename (outputs_objD_*).

#SBATCH --job-name=uni-p2-train
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00
#SBATCH --array=0-19
#SBATCH --qos=high
#SBATCH -o slurm-uni-p2-train-%A_%a.out
#SBATCH -e slurm-uni-p2-train-%A_%a.err

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

seed_list=(0 1 2 3 4 5 6 7 8 9)
backup_mode_list=(learned analytic)

reward_mode="trajectory_following"
reward_short="traj"

env_dt_val="0.05"
env_max_steps_val="400"
total_steps_val="1000000"
num_steps_val="20"
horizon_T_val="1.0"
r_max_val="1.0"

actor_lr_val="1e-4"
critic_lr_val="3e-4"
alpha_lr_val="1e-4"
min_alpha_val="1e-2"
q_clip_abs_val="5e6"

traj_y_amplitude_val="2.5"
traj_y_period_val="10.0"
traj_y_phase_val="0.0"
traj_v_mean_val="5.0"
traj_v_amplitude_val="0.0"
traj_v_period_val="10.0"
traj_v_phase_val="0.0"
traj_speed_err_scale_val="5.0"

w_v_val="20.0"
w_lane_y_val="50.0"
w_lane_psi_val="10.0"
w_control_val="0.05"

solver_tol_val="1e-4"
alpha_val="4.0"
base_alpha_val="2.0"
cbf_v_des_val="5.0"

# Unified LQR base set B (terminal set == capture set == paper base set B).
base_set_c_val="0.3"
lqr_q_y_val="1.0"
lqr_q_v_val="1.0"
lqr_q_psi_val="1.0"
lqr_r_a_val="0.01"
lqr_r_r_val="0.5"

# Learned backup policy checkpoint (staged Phase-1 unicycle safe-arrival actor).
learned_backup_policy_rel="checkpoints/deployed_sa/unicycle_sa/best_weights.pkl"
learned_backup_policy_path="${PROJECT_ROOT}/${learned_backup_policy_rel}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_seeds="${#seed_list[@]}"
num_modes="${#backup_mode_list[@]}"
total_runs=$((num_seeds * num_modes))

if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total_runs}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((total_runs - 1))]"
    exit 1
fi

idx="${task_id}"
seed_idx=$((idx % num_seeds))
idx=$((idx / num_seeds))
mode_idx=$((idx % num_modes))

seed_val="${seed_list[$seed_idx]}"
backup_mode_val="${backup_mode_list[$mode_idx]}"

if [ "${backup_mode_val}" = "analytic" ]; then
    output_mode_label="analytic"
else
    output_mode_label="learned"
    if [ ! -f "${learned_backup_policy_path}" ]; then
        echo "Learned backup policy checkpoint not found: ${learned_backup_policy_path}"
        exit 1
    fi
fi

output_root_rel="uni_phase2_ps2"
suffix="uniP2-mode_${output_mode_label}-s_${seed_val}"
tag="$(date +%Y%m%d_%H%M%S)"
uni_root="${PROJECT_ROOT}/outputs/${output_root_rel}"
run_dir="${uni_root}/${tag}-${suffix}"
tmp_log="${uni_root}/${tag}-${suffix}.train.log"
mkdir -p "${uni_root}"

exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-uni-p2-train-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-uni-p2-train-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/train_phase2.py --system unicycle
    --seed "${seed_val}"
    --total_steps "${total_steps_val}"
    --start_steps 4000
    --batch_size 64
    --update_every 8
    --gradient_steps 1
    --update_after 2000
    --hidden_size 128
    --eval_every 5000
    --eval_episodes 10
    --log_every 1000
    --max_grad_norm 5.0
    --actor_lr "${actor_lr_val}"
    --critic_lr "${critic_lr_val}"
    --alpha_lr "${alpha_lr_val}"
    --min_alpha "${min_alpha_val}"
    --q_clip_abs "${q_clip_abs_val}"
    --use_projection
    --project_target_actions
    --record_update_metrics
    --update_metric_every 200
    --reward_mode "${reward_mode}"
    --w_v "${w_v_val}"
    --w_lane_y "${w_lane_y_val}"
    --w_lane_psi "${w_lane_psi_val}"
    --w_control "${w_control_val}"
    --env_v_des 5.0
    --reward_v_des 5.0
    --traj_y_amplitude "${traj_y_amplitude_val}"
    --traj_y_period "${traj_y_period_val}"
    --traj_y_phase "${traj_y_phase_val}"
    --traj_v_mean "${traj_v_mean_val}"
    --traj_v_amplitude "${traj_v_amplitude_val}"
    --traj_v_period "${traj_v_period_val}"
    --traj_v_phase "${traj_v_phase_val}"
    --traj_normalize_reward
    --traj_speed_err_scale "${traj_speed_err_scale_val}"
    --env_dt "${env_dt_val}"
    --env_max_steps "${env_max_steps_val}"
    --r_max "${r_max_val}"
    --num_steps "${num_steps_val}"
    --horizon_T "${horizon_T_val}"
    --not_terminate_on_violation
    --alpha "${alpha_val}"
    --base_alpha "${base_alpha_val}"
    --solver_tol "${solver_tol_val}"
    --target_kappa 1e-2
    --cbf_v_des "${cbf_v_des_val}"
    --backup_policy_mode "${backup_mode_val}"
    --base_set_c "${base_set_c_val}"
    --lqr_q_y "${lqr_q_y_val}"
    --lqr_q_v "${lqr_q_v_val}"
    --lqr_q_psi "${lqr_q_psi_val}"
    --lqr_r_a "${lqr_r_a_val}"
    --lqr_r_r "${lqr_r_r_val}"
    --save_final_weights
    --output_root "${output_root_rel}"
    --output_dir "${suffix}"
    --run_tag "${tag}"
)

if [ "${backup_mode_val}" = "learned" ]; then
    cmd+=(--learned_backup_policy_path "${learned_backup_policy_path}")
fi

echo "[$(date)] Starting run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}/${total_runs}"
echo "run_dir=${run_dir}"
echo "backup_mode=${backup_mode_val} seed=${seed_val} base_set_c=${base_set_c_val}"
echo "learned_backup_policy_path=${learned_backup_policy_path}"
echo "Command: ${cmd[*]}"
echo

if [ "${DRY_RUN_ONLY:-0}" = "1" ]; then
    exit 0
fi

nvidia-smi || true
echo
"${cmd[@]}"

if [ -d "${run_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${run_dir}/train.log"
fi
echo "[$(date)] Run complete"

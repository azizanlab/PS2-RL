#!/bin/bash
# Quadrotor Phase-2 PS2 policy training (SAC through the control-invariant layer),
# comparing the learned backup policy (PS2_SA) against the analytic hybrid
# PID+dLQR backup (PS2_ABP). 10 seeds x 2 backup modes = 20 runs. Warm-started
# from the vanilla powerloop tracker. Entrypoint: scripts/train_phase2.py --system quadrotor.
#
# NOTE: output root retains the internal reproduction codename (outputs_objF_*).

#SBATCH --job-name=quad-p2-train
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00
#SBATCH --array=0-19
#SBATCH --qos=high
#SBATCH -o slurm-quad-p2-train-%A_%a.out
#SBATCH -e slurm-quad-p2-train-%A_%a.err

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
backup_policy_mode_list=(learned analytic)

env_dt_val="0.02"
env_max_steps_extra_sec_val="0.0"
total_steps_val="1500000"
start_steps_val="0"
update_after_val="1024"
num_steps_val="100"
horizon_T_val="2.0"
solver_tol_val="5e-4"
alpha_val="4.0"
base_alpha_val="2.0"

warm_start_val="true"
# Vanilla powerloop-tracker warm-start (staged).
warm_start_weights_val="checkpoints/quadrotor_vanilla/quadrotor_vanilla_weights.pkl"

actor_lr_val="5e-5"
critic_lr_val="1e-4"
alpha_lr_val="5e-5"
min_alpha_val="1e-2"
q_clip_abs_val="5e6"
best_weights_save_period_val="100000"

z_max_val="3.0"
base_set_c_val="8.0"

# Analytic hybrid PID gains
pid_kp_z_val="36.0"
pid_kv_z_val="24.0"
pid_kv_xy_val="14.0"
pid_attitude_p_gain_val="45.0"
pid_yaw_gain_scale_val="0.35"
pid_ceiling_margin_val="0.60"
pid_z_safety_gain_val="32.0"
pid_ceiling_vz_gain_val="18.0"
pid_lateral_boost_val="1.50"
pid_min_virtual_accel_z_val="0.0"

# LQR weights
lqr_q_z_val="1.0"
lqr_q_vx_val="0.16"
lqr_q_vy_val="0.16"
lqr_q_vz_val="0.4"
lqr_q_thetax_val="0.8"
lqr_q_thetay_val="0.8"
lqr_q_thetaz_val="0.16"
lqr_r_a_cmd_val="0.02"
lqr_r_omega_x_val="0.012"
lqr_r_omega_y_val="0.012"
lqr_r_omega_z_val="0.004"

w_pos_xy_val="2.5"
w_pos_z_val="2.0"
w_vel_val="4.0"
w_att_val="16.0"
w_ref_omega_x_val="0.10"
w_ref_omega_y_val="0.20"
w_ref_omega_z_val="0.05"
w_control_a_val="0.01"
w_control_omega_val="0.01"

# Learned Phase-1 backup policy (staged safe-arrival actor)
learned_backup_policy_path_val="checkpoints/deployed_sa/quadrotor_sa/best_weights.pkl"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_modes="${#backup_policy_mode_list[@]}"
num_seeds="${#seed_list[@]}"
total_runs=$((num_modes * num_seeds))

if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total_runs}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((total_runs - 1))]"
    exit 1
fi

idx="${task_id}"
seed_idx=$((idx % num_seeds))
idx=$((idx / num_seeds))
mode_idx=$((idx % num_modes))

seed_val="${seed_list[$seed_idx]}"
backup_policy_mode_val="${backup_policy_mode_list[$mode_idx]}"

warm_start_artifact="${PROJECT_ROOT}/${warm_start_weights_val}"
if [ ! -f "${warm_start_artifact}" ]; then
    echo "Warm-start checkpoint not found: ${warm_start_artifact}"
    exit 1
fi

learned_backup_policy_artifact="${PROJECT_ROOT}/${learned_backup_policy_path_val}"
if [ "${backup_policy_mode_val}" = "learned" ] && [ ! -e "${learned_backup_policy_artifact}" ]; then
    echo "Learned quadrotor backup policy artifact not found: ${learned_backup_policy_artifact}"
    exit 1
fi

output_root_rel="quad_phase2_ps2"
suffix="quadP2-mode_${backup_policy_mode_val}-s_${seed_val}"
tag="$(date +%Y%m%d_%H%M%S)"
quad_root="${PROJECT_ROOT}/outputs/${output_root_rel}"
run_dir="${quad_root}/${tag}-${suffix}"
tmp_log="${quad_root}/${tag}-${suffix}.train.log"
mkdir -p "${quad_root}"

exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-quad-p2-train-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-quad-p2-train-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/train_phase2.py --system quadrotor
    --seed "${seed_val}"
    --env_dt "${env_dt_val}"
    --env_max_steps_extra_sec "${env_max_steps_extra_sec_val}"
    --start_steps "${start_steps_val}"
    --update_after "${update_after_val}"
    --num_steps "${num_steps_val}"
    --horizon_T "${horizon_T_val}"
    --backup_policy_mode "${backup_policy_mode_val}"
    --alpha "${alpha_val}"
    --base_alpha "${base_alpha_val}"
    --solver_tol "${solver_tol_val}"
    --actor_lr "${actor_lr_val}"
    --critic_lr "${critic_lr_val}"
    --alpha_lr "${alpha_lr_val}"
    --min_alpha "${min_alpha_val}"
    --q_clip_abs "${q_clip_abs_val}"
    --project_target_actions
    --best_weights_save_period "${best_weights_save_period_val}"
    --z_max "${z_max_val}"
    --base_set_c "${base_set_c_val}"
    --pid-kp-z "${pid_kp_z_val}"
    --pid-kv-z "${pid_kv_z_val}"
    --pid-kv-xy "${pid_kv_xy_val}"
    --pid-attitude-p-gain "${pid_attitude_p_gain_val}"
    --pid-yaw-gain-scale "${pid_yaw_gain_scale_val}"
    --pid-ceiling-margin "${pid_ceiling_margin_val}"
    --pid-z-safety-gain "${pid_z_safety_gain_val}"
    --pid-ceiling-vz-gain "${pid_ceiling_vz_gain_val}"
    --pid-lateral-boost "${pid_lateral_boost_val}"
    --pid-min-virtual-accel-z "${pid_min_virtual_accel_z_val}"
    --lqr-q-z "${lqr_q_z_val}"
    --lqr-q-vx "${lqr_q_vx_val}"
    --lqr-q-vy "${lqr_q_vy_val}"
    --lqr-q-vz "${lqr_q_vz_val}"
    --lqr-q-thetax "${lqr_q_thetax_val}"
    --lqr-q-thetay "${lqr_q_thetay_val}"
    --lqr-q-thetaz "${lqr_q_thetaz_val}"
    --lqr-r-a-cmd "${lqr_r_a_cmd_val}"
    --lqr-r-omega-x "${lqr_r_omega_x_val}"
    --lqr-r-omega-y "${lqr_r_omega_y_val}"
    --lqr-r-omega-z "${lqr_r_omega_z_val}"
    --w_pos_xy "${w_pos_xy_val}"
    --w_pos_z "${w_pos_z_val}"
    --w_vel "${w_vel_val}"
    --w_att "${w_att_val}"
    --w_ref_omega_x "${w_ref_omega_x_val}"
    --w_ref_omega_y "${w_ref_omega_y_val}"
    --w_ref_omega_z "${w_ref_omega_z_val}"
    --w_control_a "${w_control_a_val}"
    --w_control_omega "${w_control_omega_val}"
    --total_steps "${total_steps_val}"
    --record_update_metrics
    --update_metric_every 200
    --eval_episodes 10
    --not_terminate_on_violation
    --save_final_weights
    --output_root "${output_root_rel}"
    --output_dir "${suffix}"
    --run_tag "${tag}"
)

if [ "${warm_start_val}" = "true" ]; then
    cmd+=(--warm_start --warm_start_weights "${warm_start_weights_val}")
fi
if [ "${backup_policy_mode_val}" = "learned" ]; then
    cmd+=(--learned_backup_policy_path "${learned_backup_policy_path_val}")
fi

echo "[$(date)] Starting run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}/${total_runs}"
echo "run_dir=${run_dir}"
echo "backup_policy_mode=${backup_policy_mode_val} seed=${seed_val} base_set_c=${base_set_c_val}"
echo "learned_backup_policy_path=${learned_backup_policy_path_val}"
echo "warm_start=${warm_start_val} warm_start_weights=${warm_start_weights_val}"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"

if [ -d "${run_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${run_dir}/train.log"
fi
echo "[$(date)] Run complete"

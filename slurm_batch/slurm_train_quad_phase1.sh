#!/bin/bash
# Quadrotor Phase-1 learned backup-policy training (discounted safe-arrival) with
# actor smoothness regularization and the discrete-time LQR base controller.
# Builds the reset library from the shipped 20 vanilla traces (seed_0, runIdx
# 0-19) and trains the safe-arrival policy. Entrypoint: scripts/train_phase1.py
# --system quadrotor.
#
# Single reproduction run (array 0-0): seed 2 / beta 0.99 reproduces the learned
# backup policy consumed by Phase-2 (quad_backup_ra-...-id2-s2-b0.99-...).
# NOTE: output root retains the internal reproduction codename (outputs_objF_*_new).

#SBATCH --job-name=quad-p1-train
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --qos=high
#SBATCH -o slurm-quad-p1-train-%A_%a.out
#SBATCH -e slurm-quad-p1-train-%A_%a.err

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

# ------------------------------ Hyperparameters ----------------------------
seed_val="2"
beta_val="0.99"
total_steps_val="5000000"
start_steps_val="5000"
update_after_val="2000"
update_every_val="8"
gradient_steps_val="1"
batch_size_val="128"
replay_size_val="400000"
hidden_size_val="128"
actor_lr_val="1e-4"
critic_lr_val="3e-4"
max_grad_norm_val="5.0"
actor_log_std_min_val="-7.0"
actor_log_std_max_val="-2.8"
eval_every_val="5000"
log_every_val="1000"
update_metric_every_val="200"
num_envs_val="32"
steps_per_jit_val="128"

tau_val="0.0025"
action_smoothness_weight_val="0.05"
exploration_std_val="0.10"
exploration_clip_val="0.10"
policy_delay_val="2"
critic_huber_delta_val="1.0"
target_policy_noise_std_val="0.0"
target_policy_noise_clip_val="0.0"

goal_mode_val="terminal"
curriculum_start_scale_val="0.0"
curriculum_increment_val="0.1"
curriculum_success_threshold_val="0.80"
curriculum_window_episodes_val="50"
curriculum_min_episodes_val="100"

horizon_T_val="2.0"
num_steps_val="100"
gravity_val="9.81"
a_cmd_min_val="0.0"
a_cmd_max_g_val="4.0"
omega_max_val="18.0"
z_max_val="3.0"
z_des_val="2.0"
alpha_val="4.0"
base_alpha_val="2.0"

base_set_c_val="8.0"
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
control_weight_val="1.0"
solver_tol_val="5e-4"
target_kappa_val="1e-2"

staged_trace_dir_val="checkpoints/quadrotor_vanilla/vanilla_traces"
max_traces_val="20"
trace_set_label_val="omega_runIdx_0to19"
velocity_perturb_max_val="1.5"
tilt_perturb_deg_max_val="30.0"
base_shell_distance_val="60"
base_shell_terminal_margin_val="66"
base_shell_region_multiplier_val="0.8"

weight_general_trace_val="1.0"
weight_near_ceiling_val="2.5"
weight_bridge_val="3.0"
weight_base_shell_val="0.75"

output_root_rel="quad_phase1_sapolicy"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
tag="$(date +%Y%m%d_%H%M%S)"
suffix="quadP1-s${seed_val}-b${beta_val}"
quad_root="${PROJECT_ROOT}/outputs/${output_root_rel}"
run_dir="${quad_root}/${tag}-${suffix}"
tmp_log="${quad_root}/${tag}-${suffix}.train.log"
mkdir -p "${quad_root}"
exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-quad-p1-train-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-quad-p1-train-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/train_phase1.py --system quadrotor
    --seed "${seed_val}"
    --total_steps "${total_steps_val}"
    --start_steps "${start_steps_val}"
    --update_after "${update_after_val}"
    --update_every "${update_every_val}"
    --gradient_steps "${gradient_steps_val}"
    --batch_size "${batch_size_val}"
    --replay_size "${replay_size_val}"
    --hidden_size "${hidden_size_val}"
    --actor_lr "${actor_lr_val}"
    --critic_lr "${critic_lr_val}"
    --max_grad_norm "${max_grad_norm_val}"
    --actor_log_std_min "${actor_log_std_min_val}"
    --actor_log_std_max "${actor_log_std_max_val}"
    --eval_every "${eval_every_val}"
    --log_every "${log_every_val}"
    --record_update_metrics
    --update_metric_every "${update_metric_every_val}"
    --num_envs "${num_envs_val}"
    --steps_per_jit "${steps_per_jit_val}"
    --beta "${beta_val}"
    --tau "${tau_val}"
    --policy_delay "${policy_delay_val}"
    --action_smoothness_weight "${action_smoothness_weight_val}"
    --exploration_std "${exploration_std_val}"
    --exploration_clip "${exploration_clip_val}"
    --target_policy_noise_std "${target_policy_noise_std_val}"
    --target_policy_noise_clip "${target_policy_noise_clip_val}"
    --critic_huber_delta "${critic_huber_delta_val}"
    --use_handoff
    --goal_mode "${goal_mode_val}"
    --collector_terminate_on_goal
    --curriculum_start_scale "${curriculum_start_scale_val}"
    --curriculum_increment "${curriculum_increment_val}"
    --curriculum_success_threshold "${curriculum_success_threshold_val}"
    --curriculum_window_episodes "${curriculum_window_episodes_val}"
    --curriculum_min_episodes "${curriculum_min_episodes_val}"

    --horizon_T "${horizon_T_val}"
    --num_steps "${num_steps_val}"
    --gravity "${gravity_val}"
    --a_cmd_min "${a_cmd_min_val}"
    --a_cmd_max_g "${a_cmd_max_g_val}"
    --omega_max "${omega_max_val}"
    --z_max "${z_max_val}"
    --z_des "${z_des_val}"
    --alpha "${alpha_val}"
    --base_alpha "${base_alpha_val}"
    --base_set_c "${base_set_c_val}"
    --lqr_q_z "${lqr_q_z_val}"
    --lqr_q_vx "${lqr_q_vx_val}"
    --lqr_q_vy "${lqr_q_vy_val}"
    --lqr_q_vz "${lqr_q_vz_val}"
    --lqr_q_thetax "${lqr_q_thetax_val}"
    --lqr_q_thetay "${lqr_q_thetay_val}"
    --lqr_q_thetaz "${lqr_q_thetaz_val}"
    --lqr_r_a_cmd "${lqr_r_a_cmd_val}"
    --lqr_r_omega_x "${lqr_r_omega_x_val}"
    --lqr_r_omega_y "${lqr_r_omega_y_val}"
    --lqr_r_omega_z "${lqr_r_omega_z_val}"
    --control_weight "${control_weight_val}"
    --solver_tol "${solver_tol_val}"
    --target_kappa "${target_kappa_val}"

    --staged_trace_dir "${staged_trace_dir_val}"
    --max_traces "${max_traces_val}"
    --trace_set_label "${trace_set_label_val}"
    --base_shell_distance "${base_shell_distance_val}"
    --base_shell_terminal_margin "${base_shell_terminal_margin_val}"
    --velocity_perturb_max "${velocity_perturb_max_val}"
    --tilt_perturb_deg_max "${tilt_perturb_deg_max_val}"
    --base_shell_region_multiplier "${base_shell_region_multiplier_val}"

    --weight_general_trace "${weight_general_trace_val}"
    --weight_near_ceiling "${weight_near_ceiling_val}"
    --weight_bridge "${weight_bridge_val}"
    --weight_base_shell "${weight_base_shell_val}"

    --output_root "${output_root_rel}"
    --output_dir "${suffix}"
    --run_tag "${tag}"
)

echo "[$(date)] Starting run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}"
echo "run_dir=${run_dir}"
echo "seed=${seed_val} beta=${beta_val} base_set_c=${base_set_c_val} total_steps=${total_steps_val} action_smoothness_weight=${action_smoothness_weight_val}"
echo "reset_library: staged_trace_dir=${staged_trace_dir_val} max_traces=${max_traces_val} velocity_perturb_max=${velocity_perturb_max_val} tilt_perturb_deg_max=${tilt_perturb_deg_max_val} base_shell_distance=${base_shell_distance_val} base_shell_terminal_margin=${base_shell_terminal_margin_val}"
echo "lqr(discrete): qz=${lqr_q_z_val} qvx=${lqr_q_vx_val} qvy=${lqr_q_vy_val} qvz=${lqr_q_vz_val} qtx=${lqr_q_thetax_val} qty=${lqr_q_thetay_val} qtz=${lqr_q_thetaz_val} ra=${lqr_r_a_cmd_val} rox=${lqr_r_omega_x_val} roy=${lqr_r_omega_y_val} roz=${lqr_r_omega_z_val}"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"

if [ -d "${run_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${run_dir}/train.log"
fi
echo "[$(date)] Run complete"

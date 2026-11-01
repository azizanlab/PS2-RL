#!/bin/bash
# Unicycle Phase-1 safe-arrival backup-policy training (discounted safe-arrival)
# + inline invariant-set comparison. Reproduces the paper's learned unicycle
# backup policy (the id48/beta=0.92 ground-truth run). Entrypoint:
# scripts/train_phase1.py --system unicycle.
#
# Single canonical run (array 0-0). The original objC driver swept 96 configs;
# beta=0.92 with the gentle-exploration / aggressive-push preset is the reported
# reproduction target (backup_ra-...-id48-s1-b0.92-...-termCap0.3).
#
# NOTE: output root retains the internal reproduction codename (outputs_objC_*).

#SBATCH --job-name=uni-p1-train
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --qos=high
#SBATCH -o slurm-uni-p1-train-%A_%a.out
#SBATCH -e slurm-uni-p1-train-%A_%a.err

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
seed_val="1"
beta_val="0.92"
total_steps_val="3000000"
start_steps_val="5000"
update_after_val="2000"
update_every_val="8"
gradient_steps_val="1"
batch_size_val="128"
hidden_size_val="128"
actor_lr_val="3e-4"
critic_lr_val="3e-4"
max_grad_norm_val="5.0"
actor_log_std_min_val="-8.0"
actor_log_std_max_val="-3.8"
eval_every_val="5000"
log_every_val="1000"
record_update_metrics_val="true"
update_metric_every_val="500"

tau_val="0.0025"
action_smoothness_weight_val="0.02"
exploration_std_val="0.08"
exploration_clip_val="0.10"
policy_delay_val="2"
critic_huber_delta_val="1.0"
target_policy_noise_std_val="0.0"
target_policy_noise_clip_val="0.0"

curriculum_start_scale_val="0.2"
curriculum_increment_val="0.005"
curriculum_success_threshold_val="0.90"
curriculum_window_episodes_val="50"
curriculum_min_episodes_val="50"

num_envs_val="32"
steps_per_jit_val="128"
val_reset_count_val="100"
test_reset_count_val="100"
collector_terminate_on_goal_val="true"
terminate_on_crash_val="true"

train_horizon_T_val="1.0"
train_num_steps_val="20"
eval_horizon_T_val="1.0"
eval_num_steps_val="20"
y_max_val="1.8"
psi_max_val="1.0471975511965976"
a_max_val="5.0"
r_max_val="1.0"
v_min_val="0.0"
v_max_val="12.0"
v_des_val="5.0"
init_y_range_min_val="0.08"
init_y_range_max_val="1.5"
init_v_range_min_val="0.1"
init_v_range_max_val="3.0"
init_psi_range_min_val="0.03"
init_psi_range_max_val="0.5"

base_set_c_val="0.3"
lqr_q_y_val="1.0"
lqr_q_v_val="1.0"
lqr_q_psi_val="1.0"
lqr_r_a_val="0.01"
lqr_r_r_val="0.5"

skip_invariant_compare_val="false"
invariant_compare_checkpoint_val="best"
compare_num_y_val="121"
compare_num_psi_val="121"
compare_num_v_val="25"
compare_v_min_val="2.0"
compare_v_max_val="8.0"
compare_max_scatter_points_val="20000"

# --------------------------------- Directory -------------------------------- #
output_root_val="uni_phase1_sapolicy"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
tag="$(date +%Y%m%d_%H%M%S)"
suffix="uniP1-s${seed_val}-b${beta_val}"
uni_root="${PROJECT_ROOT}/outputs/${output_root_val}"
run_dir="${uni_root}/${tag}-${suffix}"
tmp_log="${uni_root}/${tag}-${suffix}.train.log"
mkdir -p "${uni_root}"
exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/${USER}/mpl-uni-p1-train-${SLURM_JOB_ID:-$$}-${task_id}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg-uni-p1-train-${SLURM_JOB_ID:-$$}-${task_id}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cmd=(
    python scripts/train_phase1.py --system unicycle
    --seed "${seed_val}"
    --total_steps "${total_steps_val}"
    --start_steps "${start_steps_val}"
    --update_after "${update_after_val}"
    --update_every "${update_every_val}"
    --gradient_steps "${gradient_steps_val}"
    --batch_size "${batch_size_val}"
    --hidden_size "${hidden_size_val}"
    --actor_lr "${actor_lr_val}"
    --critic_lr "${critic_lr_val}"
    --max_grad_norm "${max_grad_norm_val}"
    --actor_log_std_min "${actor_log_std_min_val}"
    --actor_log_std_max "${actor_log_std_max_val}"
    --eval_every "${eval_every_val}"
    --log_every "${log_every_val}"
    --record_update_metrics "${record_update_metrics_val}"
    --update_metric_every "${update_metric_every_val}"
    --policy_delay "${policy_delay_val}"
    --critic_huber_delta "${critic_huber_delta_val}"
    --action_smoothness_weight "${action_smoothness_weight_val}"
    --tau "${tau_val}"
    --beta "${beta_val}"
    --exploration_std "${exploration_std_val}"
    --exploration_clip "${exploration_clip_val}"
    --target_policy_noise_std "${target_policy_noise_std_val}"
    --target_policy_noise_clip "${target_policy_noise_clip_val}"
    --num_envs "${num_envs_val}"
    --steps_per_jit "${steps_per_jit_val}"
    --val_reset_count "${val_reset_count_val}"
    --test_reset_count "${test_reset_count_val}"
    --collector_terminate_on_goal "${collector_terminate_on_goal_val}"
    --terminate_on_crash "${terminate_on_crash_val}"
    --curriculum_start_scale "${curriculum_start_scale_val}"
    --curriculum_increment "${curriculum_increment_val}"
    --curriculum_success_threshold "${curriculum_success_threshold_val}"
    --curriculum_window_episodes "${curriculum_window_episodes_val}"
    --curriculum_min_episodes "${curriculum_min_episodes_val}"

    --train_num_steps "${train_num_steps_val}"
    --train_horizon_T "${train_horizon_T_val}"
    --eval_num_steps "${eval_num_steps_val}"
    --eval_horizon_T "${eval_horizon_T_val}"
    --y_max "${y_max_val}"
    --psi_max "${psi_max_val}"
    --a_max "${a_max_val}"
    --r_max "${r_max_val}"
    --v_min "${v_min_val}"
    --v_max "${v_max_val}"
    --v_des "${v_des_val}"
    --base_set_c "${base_set_c_val}"
    --lqr_q_y "${lqr_q_y_val}"
    --lqr_q_v "${lqr_q_v_val}"
    --lqr_q_psi "${lqr_q_psi_val}"
    --lqr_r_a "${lqr_r_a_val}"
    --lqr_r_r "${lqr_r_r_val}"
    --init_y_range_min "${init_y_range_min_val}"
    --init_y_range_max "${init_y_range_max_val}"
    --init_v_range_min "${init_v_range_min_val}"
    --init_v_range_max "${init_v_range_max_val}"
    --init_psi_range_min "${init_psi_range_min_val}"
    --init_psi_range_max "${init_psi_range_max_val}"

    --compare_num_y "${compare_num_y_val}"
    --compare_num_psi "${compare_num_psi_val}"
    --compare_num_v "${compare_num_v_val}"
    --compare_v_min "${compare_v_min_val}"
    --compare_v_max "${compare_v_max_val}"
    --compare_max_scatter_points "${compare_max_scatter_points_val}"
    --invariant_compare_checkpoint "${invariant_compare_checkpoint_val}"
    --skip_invariant_compare "${skip_invariant_compare_val}"

    --output_root "${output_root_val}"
    --output_dir "${suffix}"
    --run_tag "${tag}"
)

echo "[$(date)] Starting run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}"
echo "run_dir=${run_dir}"
echo "seed=${seed_val} beta=${beta_val} base_set_c=${base_set_c_val} action_smoothness_weight=${action_smoothness_weight_val}"
echo "lqr_weights: q=(${lqr_q_y_val},${lqr_q_v_val},${lqr_q_psi_val}) r=(${lqr_r_a_val},${lqr_r_r_val})"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"

if [ -d "${run_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${run_dir}/train.log"
fi
echo "[$(date)] Run complete"

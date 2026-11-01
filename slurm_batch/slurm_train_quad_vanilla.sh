#!/bin/bash
# Vanilla quadrotor powerloop tracker training (SAC, projection OFF). Produces
# the reference-tracking policies used as the Phase-2 warm-start and as the
# source of the reset-library traces. Entrypoint: scripts/train_vanilla_tracker.py
# (which forces --disable_projection --no_project_actor_actions).
#
# Reward-hyperparameter grid (32 configs); runIdx is a grid index, not a seed.
# NOTE: output root retains the internal reproduction codename (outputs_objE_*).

#SBATCH --job-name=quad-vanilla-train
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=48:00:00
#SBATCH --array=0-31
#SBATCH -o slurm-quad-vanilla-train-%A_%a.out
#SBATCH -e slurm-quad-vanilla-train-%A_%a.err

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

env_dt_val="0.02"
total_steps_val="5000000"
reward_mode_val="trajectory_following"
seed_list=(1)

# Tracking-focused reward sweep.
p_xy_list=(2.0 2.5)
p_z_list=(2.0 3.0)
p_w_list=(4.0)
p_att_list=(16.0)
w_ref_omega_x_list=(0.05 0.10)
w_ref_omega_y_list=(0.20 0.30)
w_ref_omega_z_list=(0.05 0.10)

w_control_a_val="0.01"
w_control_omega_val="0.01"

# The env keeps the hard deck enabled internally; push it far away so it is
# irrelevant for powerloop tracking.
z_max_val="15.0"

actor_lr_val="1e-4"
critic_lr_val="5e-4"
alpha_lr_val="1e-4"
min_alpha_val="1e-1"
q_clip_abs_val="5e6"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_seeds="${#seed_list[@]}"
num_p_xy="${#p_xy_list[@]}"
num_p_z="${#p_z_list[@]}"
num_p_w="${#p_w_list[@]}"
num_p_att="${#p_att_list[@]}"
num_w_ref_omega_x="${#w_ref_omega_x_list[@]}"
num_w_ref_omega_y="${#w_ref_omega_y_list[@]}"
num_w_ref_omega_z="${#w_ref_omega_z_list[@]}"
total_runs=$((num_seeds * num_p_xy * num_p_z * num_p_w * num_p_att * num_w_ref_omega_x * num_w_ref_omega_y * num_w_ref_omega_z))

if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total_runs}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected [0, $((total_runs - 1))]"
    exit 1
fi

idx="${task_id}"
p_w_ref_omega_z_idx=$((idx % num_w_ref_omega_z))
idx=$((idx / num_w_ref_omega_z))
p_w_ref_omega_y_idx=$((idx % num_w_ref_omega_y))
idx=$((idx / num_w_ref_omega_y))
p_w_ref_omega_x_idx=$((idx % num_w_ref_omega_x))
idx=$((idx / num_w_ref_omega_x))
p_att_idx=$((idx % num_p_att))
idx=$((idx / num_p_att))
p_w_idx=$((idx % num_p_w))
idx=$((idx / num_p_w))
p_z_idx=$((idx % num_p_z))
idx=$((idx / num_p_z))
p_xy_idx=$((idx % num_p_xy))
idx=$((idx / num_p_xy))
seed_idx=$((idx % num_seeds))

seed_val="${seed_list[$seed_idx]}"
p_xy_val="${p_xy_list[$p_xy_idx]}"
p_z_val="${p_z_list[$p_z_idx]}"
p_w_val="${p_w_list[$p_w_idx]}"
p_att_val="${p_att_list[$p_att_idx]}"
w_ref_omega_x_val="${w_ref_omega_x_list[$p_w_ref_omega_x_idx]}"
w_ref_omega_y_val="${w_ref_omega_y_list[$p_w_ref_omega_y_idx]}"
w_ref_omega_z_val="${w_ref_omega_z_list[$p_w_ref_omega_z_idx]}"

output_suffix="reward_quadTrackOmega-runIdx_${task_id}-envDT_${env_dt_val}-pXY_${p_xy_val}-pZ_${p_z_val}-pW_${p_w_val}-pAtt_${p_att_val}-wOx_${w_ref_omega_x_val}-wOy_${w_ref_omega_y_val}-wOz_${w_ref_omega_z_val}-seed_${seed_val}"
tag="$(date +%Y%m%d_%H%M%S)"
output_root_val="quad_vanilla_tracker"
quad_root="${PROJECT_ROOT}/outputs/${output_root_val}"
run_dir="${quad_root}/${tag}-${output_suffix}"
tmp_log="${quad_root}/${tag}-${output_suffix}.train.log"
mkdir -p "${quad_root}"

exec > "${tmp_log}" 2>&1

module load anaconda/Python-ML-2025a

cd "${PROJECT_ROOT}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cmd=(
    python scripts/train_vanilla_tracker.py
    --seed "${seed_val}"
    --env_dt "${env_dt_val}"
    --reward_mode "${reward_mode_val}"
    --z_max "${z_max_val}"

    --actor_lr "${actor_lr_val}"
    --critic_lr "${critic_lr_val}"
    --alpha_lr "${alpha_lr_val}"
    --min_alpha "${min_alpha_val}"
    --q_clip_abs "${q_clip_abs_val}"

    --w_pos_xy "${p_xy_val}"
    --w_pos_z "${p_z_val}"
    --w_vel "${p_w_val}"
    --w_att "${p_att_val}"
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
    --output_root "${output_root_val}"
    --output_dir "${output_suffix}"
    --run_tag "${tag}"
)

echo "[$(date)] Starting run"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${task_id}"
echo "run_dir=${run_dir}"
echo "seed=${seed_val} reward_mode=${reward_mode_val} env_dt=${env_dt_val} total_steps=${total_steps_val} projection=disabled z_max=${z_max_val} p_xy=${p_xy_val} p_z=${p_z_val} p_w=${p_w_val} p_att=${p_att_val} w_ref_omega=(${w_ref_omega_x_val},${w_ref_omega_y_val},${w_ref_omega_z_val})"
echo "Command: ${cmd[*]}"
echo
nvidia-smi || true
echo
"${cmd[@]}"

mkdir -p "${quad_root}"
if [ -d "${run_dir}" ] && [ -f "${tmp_log}" ]; then
    mv "${tmp_log}" "${run_dir}/train.log"
fi
echo "[$(date)] Run complete"

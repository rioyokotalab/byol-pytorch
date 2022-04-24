#!/bin/bash

#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=8
#PJM -L elapse=24:00:00
#PJM -L node-mem=448Gi
#PJM -L proc-core=unlimited
#PJM -g jh160041a
#PJM --fs /work
#PJM --mpi proc=64
#PJM -N pretrain_byol_light_resume 
#PJM -j
#PJM -X

#------- Program execution -------#

START_TIMESTAMP=$(date '+%s')

# ======== Variables ========

job_id_base=$PJM_JOBID

git_root=$(git rev-parse --show-toplevel | head -1)

data_root="./dataset/ILSVRC2012"

log_file="$PJM_JOBNAME.$job_id_base.out"

log_root="$git_root/results_root/result/pretrain/20220417_141913"
# resume_root="$git_root/results_root/result/pretrain/20220417_141913/checkpoints_root"
# resume_root="$git_root/results_root/result/pretrain/20220417_141913/resume/20220418_143812/checkpoints_root"
# resume_root="$git_root/results_root/result/pretrain/20220417_141913/resume/20220419_145144/checkpoints_root"
# resume_root="$git_root/results_root/result/pretrain/20220417_141913/resume/20220420_150801/checkpoints_root"
resume_root="$git_root/results_root/result/pretrain/20220417_141913/resume/20220422_221054/checkpoints_root"
# resume_path="$resume_root/byol-epoch=0026-step=9450.ckpt"
last_resume_name=$(find "$resume_root" -type f | sort | tail -1)
resume_path="$last_resume_name"
# log_root="$git_root"
# resume_path~=""

epochs=1000
date_str=$(date '+%Y%m%d_%H%M%S')
result_root="$log_root/resume/$date_str"
result_path="$result_root"
# tf_logs="$result_path/lightning_logs"
git_out="$result_path/git_out"

mkdir -p "$result_path"
mkdir -p "$git_out"
# mkdir -p "$tf_logs"


# # ======== Copy ========
# 
# COPY_START_TIMESTAMP=$(date '+%s')
# 
# local_data_root="$local_ssd_path/ILSVRC2012"
# mkdir -p "$local_data_root"
# 
# rsync -avz "$data_root/$imagenet_name" "$local_data_root"
# COPY_END_TIMESTAMP=$(date '+%s')
# 
# COPY_E_TIME=$(($COPY_END_TIMESTAMP-$COPY_START_TIMESTAMP))
# echo "copy time: $COPY_E_TIME s"
# 
# export TFDS_DATA_DIR="$local_data_root"

# ======== Pyenv ========

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
which python

# ======== Modules ========

# source /etc/profile.d/modules.sh
module load gcc/8.3.1
module load cuda/11.1
# module load pytorch/1.8.1
module load cudnn/8.1.0
module load nccl/2.7.8
module load ompi/4.1.1
# # module load pytorch-horovod/1.8.1-0.21.3
# GPUS_PER_NODE=`nvidia-smi -L | wc -l`
# # source $PYTORCH_DIR/bin/activate # ← 仮想環境を activate

module list

# ======== MPI ========

nodes=$PJM_NODE
# gpus_pernode=4
# cpus_pernode=5
gpus_pernode=${PJM_PROC_BY_NODE}
# cpus_pernode=${PJM_PROC_BY_NODE}

gpus=${PJM_MPI_PROC}
# cpus=${PJM_MPI_PROC}
# cpus=$nodes
# cpus=$(($nodes * $cpus_pernode))
# gpus=$(($nodes * $gpus_pernode))

# echo "cpus: $cpus"
# echo "cpus per node $cpus_pernode"

echo "gpus: $gpus"
echo "gpus per node $gpus_pernode"

MASTER_ADDR=$(cat "$PJM_O_NODEINF" | head -1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# batch_size=32
# batch_size=64
# batch_size=128
# batch_size=256
# batch_size=512
# batch_size=1024
# batch_size=2048
batch_size=4096

# ======== Scripts ========

pushd "$git_root"

set -x

git status | tee "$git_out/git_status.txt"
git log | tee "$git_out/git_log.txt"
git diff HEAD | tee "$git_out/git_diff.txt"
git rev-parse HEAD | tee "$git_out/git_head.txt"


# mpiexec \
#     -machinefile $PJM_O_NODEINF \
#     -n $PJM_MPI_PROC \
#     -npernode $PJM_PROC_BY_NODE \

mpirun \
    -machinefile $PJM_O_NODEINF \
    -np $PJM_MPI_PROC \
    -npernode $PJM_PROC_BY_NODE \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_BUFFSIZE=1048576 \
    python examples/lightning/train.py \
    --image_folder "$data_root" \
    --result_path "$result_path" \
    --resume_path "$resume_path" \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr 3e-4 \
    --logging_set_detail \
    --imaeg_size 256

set +x

popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"

set -x

cp "$log_file" "$result_path"

set +x

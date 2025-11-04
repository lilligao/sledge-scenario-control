#!/bin/bash

# module load miniconda/3
# conda activate sledge

export PYTHONUNBUFFERED=1

export SCRATCH_ROOT="/data_nuplan/sledge_workspace"
export NUPLAN_DATA_ROOT="/data_nuplan/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data_nuplan/nuplan/dataset/maps"

export SLEDGE_EXP_ROOT="$SCRATCH_ROOT"
export SLEDGE_DEVKIT_ROOT="/mnt/efs/users/lili.gao/Repos/sledge-scenario-dreamer"


JOB_NAME=training_rvae_model
AUTOENCODER_CACHE_PATH=$SCRATCH_ROOT/caches/autoencoder_cache_scenario_control
AUTOENCODER_CHECKPOINT=null # set for weight intialization / continue training
USE_CACHE_WITHOUT_DATASET=True
SEED=0

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder_temporal.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[learning_rate_monitor_callback, model_checkpoint_callback, time_logging_callback, rvae_visualization_callback]" 
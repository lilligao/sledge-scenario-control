#!/bin/bash

# module load miniconda/3
# conda activate sledge

export PYTHONUNBUFFERED=1

export SCRATCH_ROOT="/opt/dlami/nvme/sledge_workspace"
export NUPLAN_DATA_ROOT="/data_nuplan/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data_nuplan/nuplan/dataset/maps"

export SLEDGE_EXP_ROOT="$SCRATCH_ROOT"
export SLEDGE_DEVKIT_ROOT="/your_path/sledge-scenario-control"

JOB_NAME=feature_caching_nuplan_test
AUTOENCODER_CACHE_PATH=$SCRATCH_ROOT/caches/autoencoder_cache_scenario_control
USE_CACHE_WITHOUT_DATASET=True
SEED=0

for MAP in "filter_lav_test" "filter_pgh_test" "filter_sgp_test" "filter_bos_test" 
do
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
    py_func=feature_caching \
    seed=$SEED \
    job_name=$JOB_NAME \
    +autoencoder=preprocess_rvae_temporal \
    scenario_builder=nuplan_test \
    scenario_filter=$MAP \
    cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
    cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
    callbacks="[]" \
    scenario_filter.expand_scenarios=false \
    autoencoder_model.config.filter_vehicles_by_drivable_area=False \
    autoencoder_model.config.temporal=True \
    autoencoder_model.config.sequence_length=81 \
    autoencoder_model.config.filter_non_overlapping=True \
    autoencoder_model.config.non_overlapping_gap_seconds=5.0
done

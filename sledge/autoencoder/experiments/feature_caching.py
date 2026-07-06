import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, cast
import json

from omegaconf import DictConfig
from collections import defaultdict  

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)

from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.autoencoder.experiments.feature_preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        # NOTE: Changed from nuPlan for compatibility with autoencoder wrapper
        autoencoder = build_autoencoder_torch_module_wrapper(cfg)
        feature_builders = autoencoder.get_list_of_required_feature()
        target_builders = autoencoder.get_list_of_computed_target()

        # Now that we have the feature and target builders, we do not need the model any more.
        # Delete it so it gets gc'd and we can save a few system resources.
        del autoencoder

        # Create feature preprocessor
        assert (
            cfg.cache.autoencoder_cache_path is not None
        ), f"Cache path cannot be None when caching, got {cfg.cache.autoencoder_cache_path}"
        preprocessor = FeaturePreprocessor(
            cache_path=cfg.cache.autoencoder_cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                "Processing scenario %s / %s in thread_id=%s, node_id=%s",
                idx + 1,
                len(scenarios),
                thread_id,
                node_id,
            )
            
            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)

            # Temporal feature builders return a list of per-frame features; flatten so each
            # frame is validity-checked individually (a list has no `.is_valid`).
            all_computed_features = []
            for feature in itertools.chain(features.values(), targets.values()):
                if isinstance(feature, list):
                    all_computed_features.extend(feature)
                else:
                    all_computed_features.append(feature)

            scenario_num_failures = sum(0 if feature.is_valid else 1 for feature in all_computed_features)
            scenario_num_successes = len(all_computed_features) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata

        logger.info("Finished processing scenarios for thread_id=%s, node_id=%s", thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)  # type: ignore


def _filter_scenarios_by_timestamp_custom(
    scenario_list: List[NuPlanScenario], timestamp_threshold_s: float
) -> List[NuPlanScenario]:
    """
    Filters the list of scenarios by timestamp.
    :param scenario_list: List of scenarios to filtered.
    :param timestamp_threshold_s: Threshold for filtering out scenarios clustered together in time.
    :return: Filtered list of scenarios.
    """
    if len(scenario_list) == 0:
        return scenario_list

    def _extract_initial_lidar_timestamp(scenario: NuPlanScenario) -> int:
        return cast(int, scenario._initial_lidar_timestamp)

    scenario_list.sort(key=_extract_initial_lidar_timestamp)
    filtered_scenarios = []
    min_next_timestamp = scenario_list[0]._initial_lidar_timestamp * 1e-6  # convert to seconds

    for scenario in scenario_list:
        if scenario._initial_lidar_timestamp * 1e-6 >= min_next_timestamp:
            filtered_scenarios.append(scenario)
            min_next_timestamp = scenario._initial_lidar_timestamp * 1e-6 + timestamp_threshold_s

    return filtered_scenarios


def filter_non_overlapping_scenarios(scenarios: List[AbstractScenario], min_gap_seconds) -> List[AbstractScenario]:  
    """Filter scenarios to ensure non-overlapping temporal windows."""  
      
    # Group scenarios by log name  
    log_scenarios = defaultdict(list)  
    for scenario in scenarios:  
        log_scenarios[scenario.log_name].append(scenario)  
      
    # Apply timestamp filtering to each log group  
    filtered_scenarios = []  
    for log_name, log_scenario_list in log_scenarios.items():  
        # Apply timestamp filtering  
        filtered_list = _filter_scenarios_by_timestamp_custom(log_scenario_list, min_gap_seconds)  
        filtered_scenarios.extend(filtered_list)  
      
    return filtered_scenarios


def create_sequence_token_mapping(cfg, scenarios, output_path="sequence_token_mapping.json"):  
    """  
    Create a JSON mapping from sequence_id to ordered tokens based on timestamps.  
    
    :param cfg: Omegaconf dictionary containing configuration parameters
    :param scenarios: List of NuPlanScenario objects  
    :param output_path: Path to save the JSON mapping  
    :return: Dictionary mapping sequence_id to ordered token lists  
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
    
    sequence_mapping = {}   
    for scenario_idx, scenario in enumerate(scenarios):  
        # Generate the same sequence_id as in your metadata  
        sequence_id = f"{scenario.log_name}_{scenario.token}_{scenario.start_time.time_s:.6f}"  
          
        # Get tokens (already cached)  
        tokens = scenario._lidarpc_tokens  
          
        # Skip timestamp extraction - tokens are already chronologically ordered  
        ordered_tokens = tokens   
          
        # Limit to sequence length if configured  
        if hasattr(cfg.autoencoder_model.config, 'sequence_length') and cfg.autoencoder_model.config.sequence_length:  
            sequence_length = min(cfg.autoencoder_model.config.sequence_length, len(ordered_tokens))  
            ordered_tokens = ordered_tokens[:sequence_length]  
          
        sequence_mapping[sequence_id] = {  
            'tokens': ordered_tokens,  
            'log_name': scenario.log_name,  
            'scenario_token': scenario.token,  
            'scenario_type': scenario.scenario_type,  
            'start_time': scenario.start_time.time_s,  
            'end_time': scenario.end_time.time_s,  
            'database_interval': scenario.database_interval,  
            'sequence_length': len(ordered_tokens)  
        }  
      
    # Save to JSON file  
    with open(output_path, 'w') as f:  
        json.dump(sequence_mapping, f, indent=2)  
      
    print(f"Sequence mapping saved to {output_path}")  
    print(f"Total sequences: {len(sequence_mapping)}")  
      
    return sequence_mapping


def cache_feature(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert (
        cfg.cache.autoencoder_cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.cache.autoencoder_cache_path}"

    scenario_builder = build_scenario_builder(cfg)
    if int(os.environ.get("NUM_NODES", 1)) > 1 and cfg.distribute_by_scenario:
        # Partition differently based on how the scenario builder loads the data
        repartition_strategy = scenario_builder.repartition_strategy
        if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
            scenario_filter = DistributedScenarioFilter(
                cfg=cfg,
                worker=worker,
                node_rank=int(os.environ.get("NODE_RANK", 0)),
                num_nodes=int(os.environ.get("NUM_NODES", 1)),
                synchronization_path=cfg.cache.autoencoder_cache_path,
                timeout_seconds=cfg.get("distributed_timeout_seconds", 3600),
                distributed_mode=cfg.get("distributed_mode", DistributedMode.LOG_FILE_BASED),
            )
            scenarios = scenario_filter.get_scenarios()
        elif repartition_strategy == RepartitionStrategy.INLINE:
            scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
            num_nodes = int(os.environ.get("NUM_NODES", 1))
            node_id = int(os.environ.get("NODE_RANK", 0))
            scenarios = chunk_list(scenarios, num_nodes)[node_id]
        else:
            expected_repartition_strategies = [e.value for e in RepartitionStrategy]
            raise ValueError(
                f"Expected repartition strategy to be in {expected_repartition_strategies}, got {repartition_strategy}."
            )
    else:
        logger.debug(
            "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
            "accidentally caching each scenario multiple times!"
        )
        scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)

    # Filter scenarios based on timestamp if specified
    if cfg.autoencoder_model.config.get("non_overlapping_gap_seconds", None) is not None:
        tot_scenarios = len(scenarios)
        scenarios = filter_non_overlapping_scenarios(scenarios, cfg.autoencoder_model.config.non_overlapping_gap_seconds)
        logger.info(f"Filtered scenarios from {tot_scenarios} to {len(scenarios)} based on non-overlapping gap of {cfg.autoencoder_model.config.non_overlapping_gap_seconds} seconds.")

    # Create metadata for the scenarios
    map_names = cfg.scenario_filter.get("map_names", "")
    split = cfg.scenario_builder.data_root.split('/')[-1]
    output_path = os.path.join(cfg.cache.autoencoder_cache_path, f"sequence_token_mapping_{map_names[0]}_{split}.json")
    if not os.path.exists(output_path):
        create_sequence_token_mapping(cfg, scenarios, output_path)
        logger.info(f"Created sequence token mapping at {output_path}")
    
    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    logger.info("Completed dataset caching! Failed features and targets: %s out of %s", str(num_fail), str(num_total))

    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.cache.autoencoder_cache_path), node_id, f"{map_names[0]}_{split}")
    logger.info("Done storing metadata csv file.")

from __future__ import annotations

import logging
import pathlib
import traceback
from typing import List, Optional, Tuple, Type, Union
import textwrap

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle, FeatureCacheS3, FeatureCache


logger = logging.getLogger(__name__)


def _get_temporal_cache_filename(cache_path, scenario, sequence_id, frame_idx, builder):  
    """Generate cache filename for temporal sequence frames."""  
    return (  
        cache_path / scenario.log_name / scenario.scenario_type /   
        sequence_id / f"{builder.get_feature_unique_name()}_frame_{frame_idx:04d}"  
    )  


def compute_or_load_feature(
    scenario: AbstractScenario,
    cache_path: Optional[pathlib.Path],
    builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder],
    storing_mechanism: FeatureCache,
    force_feature_computation: bool,
) -> Tuple[AbstractModelFeature, Optional[CacheMetadataEntry]]:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param cache_path: location of cached features
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :param force_feature_computation: if true, even if cache exists, it will be overwritten
    :return features computed with builder and the metadata entry for the computed feature if feature is valid.
    """
    cache_path_available = cache_path is not None

    # Filename of the cached features/targets
    file_name = (
        cache_path / scenario.log_name / scenario.scenario_type / scenario.token / builder.get_feature_unique_name()
        if cache_path_available
        else None
    )

    # Load scenario_token (scenario.token is the same as scenario.scenario_name)
    scenario_token = scenario.token

    # If feature recomputation is desired or cached file doesnt exists, compute the feature
    need_to_compute_feature = (
        force_feature_computation or not cache_path_available or not storing_mechanism.exists_feature_cache(file_name)
    )
    feature_stored_successfully = False
    if need_to_compute_feature:
        logger.debug("Computing feature...")
        if isinstance(scenario, CachedScenario):
            raise ValueError(
                textwrap.dedent(
                    f"""
                Attempting to recompute scenario with CachedScenario.
                This should typically never happen, and usually means that the scenario is missing from the cache.
                Check the cache to ensure that the scenario is present.

                If it was intended to re-compute the feature on the fly, re-run with `cache.use_cache_without_dataset=False`.

                Debug information:
                Scenario type: {scenario.scenario_type}. Scenario log name: {scenario.log_name}. Scenario token: {scenario.token}.
                """
                )
            )
        if isinstance(builder, AbstractFeatureBuilder):
            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
        else:
            raise ValueError(f"Unknown builder type: {type(builder)}")
        # If caching is enabled, store the feature
        if type(feature) is list and isinstance(builder, AbstractFeatureBuilder): 
            sequence_length = len(feature)
            for frame_idx, feature_i in enumerate(feature): 
                if feature_i.is_valid and cache_path_available:
                    file_name = _get_temporal_cache_filename(  
                        cache_path, scenario, scenario.token, frame_idx, builder  
                    ) 
                    logger.debug(f"Saving feature: {file_name} to a file...")
                    file_name.parent.mkdir(parents=True, exist_ok=True)
                    feature_stored_successfully = storing_mechanism.store_computed_feature_to_folder(file_name, feature_i)
        elif isinstance(builder, AbstractTargetBuilder):
            file_name = (  
                cache_path / scenario.log_name / scenario.scenario_type /   
                scenario.token / builder.get_feature_unique_name()  
            ) 
            logger.debug(f"Saving map id: {file_name} to a file...")
            file_name.parent.mkdir(parents=True, exist_ok=True)
            feature_stored_successfully = storing_mechanism.store_computed_feature_to_folder(file_name, feature) 
        elif feature.is_valid and cache_path_available:
            logger.debug(f"Saving feature: {file_name} to a file...")
            file_name.parent.mkdir(parents=True, exist_ok=True)
            feature_stored_successfully = storing_mechanism.store_computed_feature_to_folder(file_name, feature) 
    else:
        # In case the feature exists in the cache, load it
        logger.debug(f"Loading feature: {file_name} from a file...")
        feature = storing_mechanism.load_computed_feature_from_folder(file_name, builder.get_feature_type())
        assert feature.is_valid, 'Invalid feature loaded from cache!'

    return (
        feature,
        CacheMetadataEntry(file_name=file_name, scenario_token=scenario_token) if (need_to_compute_feature and feature_stored_successfully) else None,
    )


class FeaturePreprocessor:
    """
    Compute features and targets for a scenario. This class also manages cache. If a feature/target
    is not present in a cache, it is computed, otherwise it is loaded
    """

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        :param feature_builders: List of feature builders.
        :param target_builders: List of target builders.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._storing_mechanism = (
            FeatureCacheS3(cache_path) if str(cache_path).startswith('s3://') else FeatureCachePickle()
        )

        assert len(feature_builders) != 0, "Number of feature builders has to be greater than 0!"

    @property
    def feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: all feature builders
        """
        return self._feature_builders

    @property
    def target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: all target builders
        """
        return self._target_builders

    def get_list_of_feature_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all features that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._feature_builders]

    def get_list_of_target_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all targets that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._target_builders]

    def compute_features(
        self, scenario: AbstractScenario
    ) -> Tuple[FeaturesType, TargetsType, List[CacheMetadataEntry]]:
        """
        Compute features for a scenario, in case cache_path is set, features will be stored in cache,
        otherwise just recomputed
        :param scenario for which features and targets should be computed
        :return: model features and targets and cache metadata
        """
        try:
            all_features: FeaturesType
            all_feature_cache_metadata: List[CacheMetadataEntry]
            all_targets: TargetsType
            all_targets_cache_metadata: List[CacheMetadataEntry]

            all_features, all_feature_cache_metadata = self._compute_all_features(scenario, self._feature_builders)
            all_targets, all_targets_cache_metadata = self._compute_all_features(scenario, self._target_builders)

            all_cache_metadata = all_feature_cache_metadata + all_targets_cache_metadata
            return all_features, all_targets, all_cache_metadata
        except Exception as error:
            msg = (
                f"Failed to compute features for scenario token {scenario.token} in log {scenario.log_name}\n"
                f"Error: {error}"
            )

            logger.error(msg)
            traceback.print_exc()
            raise RuntimeError(msg)

    def _compute_all_features(
        self, scenario: AbstractScenario, builders: List[Union[AbstractFeatureBuilder, AbstractTargetBuilder]]
    ) -> Tuple[Union[FeaturesType, TargetsType], List[Optional[CacheMetadataEntry]]]:
        """
        Compute all features/targets from builders for scenario
        :param scenario: for which features should be computed
        :param builders: to use for feature computation
        :return: computed features/targets and the metadata entries for the computed features/targets
        """
        # Features to be computed
        all_features: FeaturesType = {}
        all_features_metadata_entries: List[CacheMetadataEntry] = []

        for builder in builders:
            feature, feature_metadata_entry = compute_or_load_feature(
                scenario, self._cache_path, builder, self._storing_mechanism, self._force_feature_computation
            )
            all_features[builder.get_feature_unique_name()] = feature
            all_features_metadata_entries.append(feature_metadata_entry)

        return all_features, all_features_metadata_entries

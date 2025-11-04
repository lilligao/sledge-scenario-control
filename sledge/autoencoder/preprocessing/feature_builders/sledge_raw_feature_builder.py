from typing import List, Type, Union, Tuple, Dict, Any
from shapely.geometry import Polygon
import pickle
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.database.nuplan_db.query_session import execute_one, execute_many
from pyquaternion import Quaternion

from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw, SledgeConfig, SledgeVectorRawNew
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_agent_feature import (
    compute_ego_features,
    compute_agent_features,
    compute_static_object_features,
)
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_line_feature import (
    compute_line_features,
    compute_traffic_light_features,
)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import CameraChannel


class SledgeRawFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder object for raw vector representation to train autoencoder in sledge"""

    def __init__(self, config: SledgeConfig):
        """
        Initializes the feature builder.
        :param config: configuration dataclass of autoencoder in sledge.
        """
        self._config = config

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "sledge_raw"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return SledgeVectorRawNew

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> SledgeVectorRaw:
        """Inherited, see superclass."""

        history = current_input.history
        ego_state = history.ego_states[-1]
        detections = history.observations[-1]
        map_api = initialization.map_api
        traffic_light_data = current_input.traffic_light_data

        return self._compute_features(ego_state, map_api, detections, traffic_light_data)

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Union[SledgeVectorRawNew, List[SledgeVectorRawNew]]:  
        """Inherited, see superclass."""  
        
        if self._config.temporal:  
            return self._get_temporal_features_from_scenario(scenario)  
        else:  
            return self._get_single_frame_features_from_scenario(scenario, iteration=0)  
    
    def _get_single_frame_features_from_scenario(self, scenario: AbstractScenario, iteration: int = 0) -> SledgeVectorRawNew:  
        """Extract features for a single iteration."""  
        
        # Get data for specific iteration  
        ego_state = scenario.get_ego_state_at_iteration(iteration)  
        detections = scenario.get_tracked_objects_at_iteration(iteration)  
        map_api = scenario.map_api  
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))  
    
        log_file = scenario._log_file  
        tokens = scenario._lidarpc_tokens  
        token = tokens[iteration]  

        #  Create comprehensive metadata  
        metadata = self._create_frame_metadata(scenario, iteration, token)  
        
        # Your existing database queries  
        ego_z, ego_rotation = self._extract_ego_data(token, log_file)  
        list_objs_z = self._extract_tracked_objects_data(token, log_file)  
        cam_infos = self._extract_camera_data([token], log_file)  
        
        return self._compute_features(ego_state, map_api, detections, traffic_light_data, ego_z, ego_rotation, list_objs_z, cam_infos, metadata)
    
    def _get_temporal_features_from_scenario(self, scenario: AbstractScenario) -> List[SledgeVectorRawNew]:  
        """Extract features for all iterations in temporal sequence."""  
        
        num_iterations = scenario.get_number_of_iterations()  
        sequence_length = getattr(self._config, 'sequence_length', num_iterations)  
        
        # Limit to available iterations or configured sequence length  
        actual_length = min(sequence_length, num_iterations)  
        
        temporal_features = []  
        
        for iteration in range(actual_length):  
            frame_features = self._get_single_frame_features_from_scenario(scenario, iteration)  
            temporal_features.append(frame_features)  
        
        return temporal_features  

    def _create_frame_metadata(self, scenario: AbstractScenario, iteration: int, token: str) -> Dict[str, Any]:  
        """Create comprehensive metadata for a single frame."""  
        
        # Get current iteration timestamp  
        current_time_point = scenario.get_time_point(iteration)  

        # Get total number of frames in the scenario (sequence_length is None => use full scenario)
        num_iterations = scenario.get_number_of_iterations()
        sequence_length = getattr(self._config, 'sequence_length', None)
        total_frames = min(num_iterations, sequence_length) if sequence_length else num_iterations
        
        # Basic scenario information  
        metadata = {  
            # Core identifiers  
            'log_name': scenario.log_name,  
            'scenario_token': scenario.token,  
            'scenario_name': scenario.scenario_name,  # Same as token in nuPlan  
            'scenario_type': scenario.scenario_type,  
            'lidar_token': token,  
            
            # Temporal information  
            'start_time': scenario.start_time.time_s,  
            'end_time': scenario.end_time.time_s,  
            'duration_s': scenario.duration_s.time_s,  
            'current_iteration': iteration,
            'database_interval': scenario.database_interval,  
            
            # Frame-specific information  
            'iteration': iteration,  
            'sequence_length': total_frames,
            'frame_progress': iteration / max(1, total_frames - 1),  # Avoid division by zero
            
            # Map information  
            'map_name': scenario._map_name if hasattr(scenario, '_map_name') else None,  
            'map_version': scenario._map_version if hasattr(scenario, '_map_version') else None,  
            
            # Vehicle information  
            'ego_vehicle_parameters': {  
                'length': scenario.ego_vehicle_parameters.length,  
                'width': scenario.ego_vehicle_parameters.width,  
                'height': scenario.ego_vehicle_parameters.height,  
                'wheel_base': scenario.ego_vehicle_parameters.wheel_base,  
            } if scenario.ego_vehicle_parameters else None,  
        }  
        
        # Add temporal context if available  
        if scenario.get_number_of_iterations() > 1:  
            metadata.update({  
                'time_from_start': current_time_point.time_s - scenario.start_time.time_s,  
                'time_to_end': scenario.end_time.time_s - current_time_point.time_s,  
                'is_first_frame': iteration == 0,  
                'is_last_frame': iteration == total_frames - 1,
            })  
        
        # Add sequence information if in temporal mode  
        if hasattr(self._config, 'temporal') and self._config.temporal:  
            sequence_id = f"{scenario.log_name}_{scenario.token}_{scenario.start_time.time_s:.6f}"  
            metadata.update({  
                'sequence_id': sequence_id,  
                'is_temporal_sequence': True,  
                'sequence_length': getattr(self._config, 'sequence_length', scenario.get_number_of_iterations()),  
            })  
        else:  
            metadata.update({  
                'is_temporal_sequence': False,  
            })  
        
        # Add mission goal if available  
        try:  
            mission_goal = scenario.get_mission_goal()  
            if mission_goal:  
                metadata['mission_goal'] = {  
                    'x': mission_goal.x,  
                    'y': mission_goal.y,  
                    'heading': mission_goal.heading,  
                }  
        except Exception:  
            metadata['mission_goal'] = None  
        
        # Add route information if available  
        try:  
            route_roadblock_ids = scenario.get_route_roadblock_ids()  
            metadata['route_roadblock_ids'] = route_roadblock_ids  
            metadata['num_route_roadblocks'] = len(route_roadblock_ids) if route_roadblock_ids else 0  
        except Exception:  
            metadata['route_roadblock_ids'] = None  
            metadata['num_route_roadblocks'] = 0  
        
        return metadata
    
    def _extract_ego_data(self, token: str, log_file) -> Tuple[float, Quaternion]:  
        """Extract ego pose data for a specific token."""  
        query_ego = """  
            SELECT  ep.z, ep.qw, ep.qx, ep.qy, ep.qz  
            FROM ego_pose AS ep  
            INNER JOIN lidar_pc AS lp ON lp.ego_pose_token = ep.token  
            WHERE lp.token = ?  
        """  
        
        row_ego = execute_one(query_ego, (bytearray.fromhex(token),), log_file)  
        ego_z = row_ego["z"]  
        ego_rotation = Quaternion(row_ego["qw"], row_ego["qx"], row_ego["qy"], row_ego["qz"])  
        
        return ego_z, ego_rotation  
    
    def _extract_tracked_objects_data(self, token: str, log_file) -> Dict[str, float]:  
        """Extract tracked objects data for a specific token."""  
        query_tracked_objects = """  
            SELECT  lb.z, lb.track_token  
            FROM lidar_box AS lb  
            INNER JOIN track AS t ON t.token = lb.track_token  
            INNER JOIN category AS c ON c.token = t.category_token  
            INNER JOIN lidar_pc AS lp ON lp.token = lb.lidar_pc_token  
            WHERE lp.token = ?  
        """  
        
        list_objs_z = {}  
        for row_obj in execute_many(query_tracked_objects, (bytearray.fromhex(token),), log_file):  
            track_token = row_obj["track_token"].hex() if row_obj["track_token"] is not None else None
            if track_token is None:
                continue
            list_objs_z[track_token] = row_obj["z"] 
        
        return list_objs_z  
    
    def _extract_camera_data(self, tokens: List[str], log_file) -> Dict[str, Dict]:  
        """Extract camera data for given tokens."""  
        channels = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']  
        
        query_cam_image = f"""  
        SELECT  
                img.token, img.next_token, img.prev_token, img.ego_pose_token,  
                img.camera_token, img.filename_jpg, img.timestamp,  
                cam.width, cam.height, cam.model, cam.intrinsic,  
                cam.translation, cam.rotation, cam.distortion,  
                cam.log_token, cam.channel,  
                ep.x AS ep_x, ep.y AS ep_y, ep.z AS ep_z,  
                ep.qw AS ep_qw, ep.qx AS ep_qx, ep.qy AS ep_qy, ep.qz AS ep_qz  
            FROM image AS img  
            INNER JOIN lidar_pc AS lpc  
                ON  img.timestamp <= lpc.timestamp + ?  
                AND img.timestamp >= lpc.timestamp - ?  
            INNER JOIN camera AS cam ON cam.token = img.camera_token  
            INNER JOIN ego_pose AS ep ON ep.token = img.ego_pose_token  
            WHERE cam.channel IN ({('?,'*len(channels))[:-1]})   
            AND lpc.token IN ({('?,'*len(tokens))[:-1]})  
            ORDER BY lpc.timestamp ASC;  
        """  
        
        lookahead_window_us = 50000   
        lookback_window_us = 50000  
        args = [lookahead_window_us, lookback_window_us]  
        args += channels  
        args += [bytearray.fromhex(t) for t in tokens]  
    
        cam_infos = {}  
        for row in execute_many(query_cam_image, args, log_file):  
            channel = row["channel"]  
            if channel not in cam_infos and row["token"] is not None:  
                cam_infos[channel] = {  
                    'token': row["token"].hex(),  
                    'next_token': row["next_token"].hex() if row["next_token"] is not None else None,  
                    'prev_token': row["prev_token"].hex() if row["prev_token"] is not None else None,  
                    'ego_pose_token': row["ego_pose_token"].hex() if row["ego_pose_token"] is not None else None,  
                    'camera_token': row["camera_token"].hex() if row["camera_token"] is not None else None,  
                    'filename_jpg': row["filename_jpg"],  
                    'timestamp': row["timestamp"],  
                    'width': row["width"],  
                    'height': row["height"],  
                    'model': row["model"],  
                    'intrinsic': list(pickle.loads(row["intrinsic"])) if row["intrinsic"] is not None else None,  
                    'translation': list(pickle.loads(row["translation"])) if row["translation"] is not None else None,  
                    'rotation': list(pickle.loads(row["rotation"])) if row["rotation"] is not None else None,  
                    'distortion': list(pickle.loads(row["distortion"])) if row["distortion"] is not None else None,  
                    'log_token': row["log_token"].hex() if row["log_token"] is not None else None,  
                    'ego_pose': [row["ep_x"], row["ep_y"], row["ep_z"],   
                            row["ep_qw"], row["ep_qx"], row["ep_qy"], row["ep_qz"]]  
                }  
        
        return cam_infos

    def _compute_features(
        self,
        ego_state,
        map_api,
        detections,
        traffic_light_data,
        ego_z,
        ego_rotation,
        list_objs_z,
        cam_infos,
        metadata
    ):
        """
        Compute raw vector feature for autoencoder training.
        :param ego_state: object of ego vehicle state in nuPlan
        :param map_api: object of map in nuPlan
        :param detections: dataclass of detected objects in scenario
        :param traffic_light_data: dataclass of traffic lights in nuPlan
        :return: raw vector representation of lines, agents, objects, and lane graph etc.
        """

        drivable_area_map = get_drivable_area_map(map_api, ego_state, self._config.radius)

        lines, G = compute_line_features(
            ego_state,
            map_api,
            self._config.radius,
            self._config.pose_interval,
        )
        # TODO: mark vehicles in drivable area or not instead of filter
        vehicles = compute_agent_features(
            ego_state,
            detections,
            TrackedObjectType.VEHICLE,
            self._config.radius,
            list_objs_z = list_objs_z,
            ego_z = ego_z,
            ego_rotation = ego_rotation
        )
        pedestrians = compute_agent_features(
            ego_state,
            detections,
            TrackedObjectType.PEDESTRIAN,
            self._config.radius,
            list_objs_z = list_objs_z,
            ego_z = ego_z,
            ego_rotation = ego_rotation
        )
        static_objects = compute_static_object_features(
            ego_state,
            detections,
            self._config.radius,
            drivable_area_map if self._config.filter_vehicles_by_drivable_area else None,
            list_objs_z = list_objs_z,
            ego_z = ego_z,
            ego_rotation = ego_rotation
        )
        green_lights = compute_traffic_light_features(
            ego_state,
            map_api,
            traffic_light_data,
            TrafficLightStatusType.GREEN,
            self._config.radius,
            self._config.pose_interval,
        )
        red_lights = compute_traffic_light_features(
            ego_state,
            map_api,
            traffic_light_data,
            TrafficLightStatusType.RED,
            self._config.radius,
            self._config.pose_interval,
        )
        ego = compute_ego_features(ego_state, ego_z, ego_rotation)

        return SledgeVectorRawNew(lines, vehicles, pedestrians, static_objects, green_lights, red_lights, ego, G, cam_infos, metadata)


def get_drivable_area_map(
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float,
    map_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION,
    ],
) -> PDMOccupancyMap:
    """
    Helper function to create occupancy map of road polygons.
    :param map_api: object of map in nuPlan
    :param ego_state: object of ego vehicle state in nuPlan
    :param map_radius: radius around the ego vehicle to load polygons
    :param map_layers: layers to load, defaults to [ SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, ]
    :return: occupancy map from PDM
    """
    # query all drivable map elements around ego position
    drivable_area = map_api.get_proximal_map_objects(ego_state.center.point, map_radius, map_layers)

    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []

    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION]:
        for layer_object in drivable_area[layer]:
            drivable_polygons.append(layer_object.polygon)
            drivable_polygon_ids.append(layer_object.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map

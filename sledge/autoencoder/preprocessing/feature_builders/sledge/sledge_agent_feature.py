from typing import List, Optional, Dict

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array, convert_absolute_to_relative_heading
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import state_se2_to_array
from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from sledge.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    StaticObjectIndex,
    AgentIndex,
    EgoIndex,
)
from pyquaternion import Quaternion


# ---------- helpers ----------
def _get_box_center_z(agent: Agent, list_objs_z: Optional[Dict[str, float]]) -> Optional[float]:
    """
    Get the agent's box center z robustly:
    prefer agent.box.center.z → agent.box.z → list_objs_z[track_token] → None
    """
    try:
        return float(agent.box.center.z)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return float(agent.box.z)  # type: ignore[attr-defined]
    except Exception:
        pass
    if list_objs_z is not None:
        z = list_objs_z.get(agent.track_token)
        if z is not None:
            return float(z)
    return None

def _relative_xyz_in_ego_frame(
    ego_xyz: npt.NDArray[np.float64],
    ego_quat: Quaternion,
    pts_xyz: npt.NDArray[np.float64],        # shape (..., 3)
) -> npt.NDArray[np.float64]:
    """
    Translate global XYZ by ego origin and rotate by inverse ego quaternion.
    Returns (..., 3) in ego frame.
    """
    rel = pts_xyz - ego_xyz  # broadcast
    flat = rel.reshape(-1, 3)
    flat_rot = np.stack([ego_quat.inverse.rotate(v) for v in flat], axis=0)
    return flat_rot.reshape(rel.shape)


def compute_ego_features(ego_state: EgoState, ego_z: float, ego_rotation: Quaternion) -> SledgeVectorElement:
    """
    Compute raw sledge vector features for ego agents
    :param ego_state: object of ego vehicle state in nuPlan
    :return: sledge vector element of raw ego attributes.
    """

    state_array = ego_state_to_state_array(ego_state)
    ego_states = np.zeros((EgoIndex.size()+9), dtype=np.float64)
    ego_mask = np.ones((1), dtype=bool)  # dummy value
    ego_states[EgoIndex.VELOCITY_2D] = state_array[StateIndex.VELOCITY_2D]
    ego_states[EgoIndex.ACCELERATION_2D] = state_array[StateIndex.ACCELERATION_2D]
    ego_states[4] = ego_state.center.x
    ego_states[5] = ego_state.center.y
    ego_states[6] = ego_state.center.heading
    ego_states[7] = ego_z  # z coordinate of the ego vehicle
    ego_states[8] = ego_state.agent.box.height  # height of the ego vehicle
    ego_states[9] = ego_rotation.w
    ego_states[10] = ego_rotation.x
    ego_states[11] = ego_rotation.y
    ego_states[12] = ego_rotation.z
    return SledgeVectorElement(ego_states, ego_mask)


# TODO: Refactor
def compute_agent_features(
    ego_state: EgoState,
    detections: DetectionsTracks,
    agent_type: TrackedObjectType,
    radius: float,
    drivable_area_map: Optional[PDMOccupancyMap] = None,
    list_objs_z: Optional[Dict[str, float]] = None,
    ego_z: Optional[float] = None,
    ego_rotation: Optional[Quaternion] = None,
) -> SledgeVectorElement:
    """
    Computes raw sledge vector features for agents (ie. vehicles, pedestrians)
    :param ego_state: object of ego vehicle state in nuPlan
    :param detections: dataclass for detected objects in nuPlan
    :param agent_type: enum of agent type (ie. vehicles, pedestrians)
    :param radius: radius around the ego vehicle to extract objects
    :param drivable_area_map: drivable area map for filtering if provided, defaults to None
    :return: raw sledge vector element of agent_type
    """

    tracked_objects = detections.tracked_objects
    agents_list: List[Agent] = tracked_objects.get_tracked_objects_of_type(agent_type)

    if ego_z is None or ego_rotation is None:
        raise ValueError("ego_z and ego_rotatio nmust be provided to compute relative z.")
    # Indices for appended features (avoid magic numbers)
    z_idx = AgentIndex.size()
    h_idx = AgentIndex.size() + 1
    agents_states_list: List[npt.NDArray[np.float64]] = []
    for agent in agents_list:
        agent_states_ = np.zeros(AgentIndex.size()+2, dtype=np.float64)
        agent_states_[AgentIndex.STATE_SE2] = state_se2_to_array(agent.center)
        agent_states_[AgentIndex.WIDTH] = agent.box.width
        agent_states_[AgentIndex.LENGTH] = agent.box.length
        agent_states_[AgentIndex.VELOCITY] = agent.velocity.magnitude()
        # --- proper ego-frame z (SE3) + height ---
        agent_states_[h_idx] = float(agent.box.height)
        agent_states_[z_idx] = _get_box_center_z(agent, list_objs_z)

        agents_states_list.append(agent_states_)
    agents_states_all = np.array(agents_states_list)

    # convert to local coords and filter out of box
    if len(agents_states_all) > 0:
        if drivable_area_map is not None:
            in_drivable_area = drivable_area_map.points_in_polygons(agents_states_all[..., AgentIndex.POINT]).any(
                axis=0
            )
            agents_states_all = agents_states_all[in_drivable_area]

        # convert to local coordinates
        ego_xyz = np.array([ego_state.center.x, ego_state.center.y, float(ego_z)], dtype=np.float64)
        agents_states_all[..., [AgentIndex.X, AgentIndex.Y, z_idx]] = _relative_xyz_in_ego_frame(ego_xyz, ego_rotation, agents_states_all[..., [AgentIndex.X, AgentIndex.Y, z_idx]])
        agents_states_all[..., AgentIndex.STATE_SE2] = convert_absolute_to_relative_heading(
            ego_state.center, agents_states_all[..., AgentIndex.STATE_SE2]
        )

        # filter detections
        within_radius = np.linalg.norm(agents_states_all[..., AgentIndex.POINT], axis=-1) <= radius
        agents_states_all = agents_states_all[within_radius]

    agent_states = np.array(agents_states_all, dtype=np.float32)
    agent_mask = np.zeros(len(agents_states_all), dtype=bool)

    return SledgeVectorElement(agent_states, agent_mask)


def compute_static_object_features(
    ego_state: EgoState,
    detections: DetectionsTracks,
    radius: float,
    drivable_area_map: Optional[PDMOccupancyMap] = None,
    list_objs_z: Optional[Dict[str, float]] = None,
    ego_z: Optional[float] = None,
    ego_rotation: Optional[Quaternion] = None,
) -> SledgeVectorElement:
    """
    Computes raw sledge vector features for static objects (ie. barriers, generic)
    :param ego_state: object of ego vehicle state in nuPlan
    :param detections: dataclass for detected objects in nuPlan
    :param radius: radius around the ego vehicle to extract objects
    :param drivable_area_map: drivable area map for filtering if provided, defaults to None
    :return: raw sledge vector element of all static object classes
    """

    tracked_objects = detections.tracked_objects
    objects_list: List[Agent] = tracked_objects.get_static_objects()

    if ego_z is None or ego_rotation is None:
        raise ValueError("ego_z and ego_rotation must be provided to compute relative z.")
    # Indices for appended features (avoid magic numbers)
    z_idx = StaticObjectIndex.size()
    h_idx = StaticObjectIndex.size() + 1
    objects_states_list: List[npt.NDArray[np.float64]] = []
    for object in objects_list:
        object_states_ = np.zeros(StaticObjectIndex.size()+2, dtype=np.float64)
        object_states_[StaticObjectIndex.STATE_SE2] = state_se2_to_array(object.center)
        object_states_[StaticObjectIndex.WIDTH] = object.box.width
        object_states_[StaticObjectIndex.LENGTH] = object.box.length
        # --- proper ego-frame z (SE3) + height ---
        object_states_[h_idx] = float(object.box.height)
        object_states_[z_idx] = _get_box_center_z(object, list_objs_z)
        objects_states_list.append(object_states_)
    objects_states_all = np.array(objects_states_list)

    # convert to local coords and filter out of box
    if len(objects_states_all) > 0:
        if drivable_area_map is not None:
            in_drivable_area = drivable_area_map.points_in_polygons(
                objects_states_all[..., StaticObjectIndex.POINT]
            ).any(axis=0)
            objects_states_all = objects_states_all[in_drivable_area]

        # convert to local coordinates
        ego_xyz = np.array([ego_state.center.x, ego_state.center.y, float(ego_z)], dtype=np.float64)
        objects_states_all[..., [StaticObjectIndex.X, StaticObjectIndex.Y, z_idx]] = _relative_xyz_in_ego_frame(ego_xyz, ego_rotation, objects_states_all[..., [StaticObjectIndex.X, StaticObjectIndex.Y, z_idx]])
        objects_states_all[..., StaticObjectIndex.STATE_SE2] = convert_absolute_to_relative_heading(
            ego_state.center, objects_states_all[..., StaticObjectIndex.STATE_SE2]
        )
        # filter detections
        within_radius = np.linalg.norm(objects_states_all[..., StaticObjectIndex.POINT], axis=-1) <= radius
        objects_states_all = objects_states_all[within_radius]

    object_states = np.array(objects_states_all, dtype=np.float32)
    object_mask = np.zeros(len(objects_states_all), dtype=bool)

    return SledgeVectorElement(object_states, object_mask)
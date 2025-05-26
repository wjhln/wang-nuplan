from typing import Dict
import numpy as np
from jedi.inference.gradual.typing import Tuple
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.script.run_simulation import logger


def route_roadblock_correction(
        ego_state: EgoState,
        route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
        search_depth_forward: int = 15,
        search_depth_backward: int = 30,
):
    pass


def get_current_roadblock_candidates(
        ego_state: EgoState,
        map_api: AbstractMap,
        route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
        heading_error_threshold: float = np.pi / 4,
        displacement_error_threshold: float = 3.0,
) -> Tuple[RoadBlockGraphEdgeMapObject, RoadBlockGraphEdgeMapObject]:

    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(ego_state.rear_axle.point, 1.0, layers)
    roadblock_candidate = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK] + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )
    if not roadblock_candidate:
        logger.warning("No roadblock candidates found in the vicinity of the ego vehicle.")

    for roadblock in roadblock_candidate:
        lane_displacement_error = np.inf
        for lane in roadblock.interior_edges:
            nearest_state = lane.baseline_path.get_nearest_pose_from_position(ego_state.rear_axle.point)
            distance_diff = nearest_state.distance_to(ego_state.rear_axle)
            heading_diff = normalize_angle(nearest_state.heading - ego_state.rear_axle.heading)

            if distance_diff < displacement_error_threshold and abs(heading_diff) < heading_error_threshold:
                if distance_diff < lane_displacement_error:
                    lane_displacement_error = distance_diff
                    current_lane = lane


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
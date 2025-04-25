from typing import Type, List, Optional
import numpy as np
from loguru import logger
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class PDMOpenPlanner(AbstractPlanner):
    def __init__(
        self,
        map_radius: float = 50.0,
    ):
        self.count = 0
        self.vehicle = get_pacifica_parameters()
        self.model = KinematicBicycleModel(self.vehicle)
        self.map_radius = map_radius

        self.map_api: Optional[AbstractMap] = None
        self.miss_goal: Optional[StateSE2] = None
        self.route_roadblock_ids : List[str] = []
    def name(self):
        return self.__class__.__name__
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        self.map_api = initialization.map_api
        self.miss_goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        logger.warning(f"miss_goal x:{self.miss_goal.x}, y:{self.miss_goal.y}")
        pass

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        logger.info(f"PDM Open Planner: {self.count}")

        ego_state, _ = current_input.history.current_state
        logger.info(f"ego_state x:{ego_state.rear_axle.x}, y:{ego_state.rear_axle.y}x")
        layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        roadblock_dict = self.map_api.get_proximal_map_objects(ego_state.rear_axle.point, self.map_radius, layers)
        roadblock_candidate = (roadblock_dict[SemanticMapLayer.ROADBLOCK] + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR])

        start_roadblock = None
        current_lane = None
        distance_diff_threshold = 3.0
        heading_diff_threshold = np.pi / 4
        distance_diff_thresh = np.inf
        for roadblock in roadblock_candidate:
            for lane in roadblock.interior_edges:
                nearest_state = lane.baseline_path.get_nearest_pose_from_position(ego_state.rear_axle.point)
                distance_diff = nearest_state.distance_to(ego_state.rear_axle)
                heading_diff = nearest_state.heading - ego_state.rear_axle.heading
                if abs(distance_diff) < distance_diff_threshold and abs(heading_diff) < heading_diff_threshold:
                    if abs(distance_diff) < distance_diff_thresh:
                        distance_diff_thresh = abs(distance_diff)
                        start_roadblock = roadblock
                        current_lane = lane

        logger.info(f"roadblock: {start_roadblock.id}")

        for state in current_lane.baseline_path.discrete_path:
            logger.info(f"lane x: {state.x}  y: {state.y}  heading: {state.heading}")



        ego_state, _ = current_input.history.current_state
        self.count += 1
        trajectory = [ego_state]
        next_state = EgoState.build_from_rear_axle(
                rear_axle_pose=ego_state.rear_axle,
                rear_axle_velocity_2d=StateVector2D(0.1, 0),
                rear_axle_acceleration_2d=StateVector2D(0, 0),
                tire_steering_angle=0,
                time_point=ego_state.time_point,
                vehicle_parameters=ego_state.car_footprint.vehicle_parameters
            )
        for i in range(1, 20):
            next_state = self.model.propagate_state(next_state, next_state.dynamic_car_state, TimePoint(int(0.25 * 1e6)))
            trajectory.append(next_state)
        return InterpolatedTrajectory(trajectory)

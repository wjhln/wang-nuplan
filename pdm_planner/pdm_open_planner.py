from abc import ABC
from typing import Optional, List, Type
from loguru import logger
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import TimePoint, StateSE2, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


class PDMOpenPlanner(AbstractPlanner):
    def __init__(self):
        self.map_api: Optional[AbstractMap] = None
        self._map_radius: int = 0
        self.count = 0
        
        # 获取Pacifica车辆参数
        self.vehicle = get_pacifica_parameters()
        self.model = KinematicBicycleModel(self.vehicle)

    def name(self) -> str:
        return self.__class__.__name__
    
    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        pass

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
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
        logger.info(f"计算规划轨迹完成，当前计数：{self.count}")
        return InterpolatedTrajectory(trajectory)
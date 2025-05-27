import gc
import os
from typing import Type, List, Optional, Dict
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.script.run_simulation import logger
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
import matplotlib.pyplot as plt  # 添加导入matplotlib


class PDMOpenPlanner(AbstractPlanner):
    def __init__(
            self,
            map_radius: float = 50.0,
    ):
        self.count = 0
        self.vehicle = get_pacifica_parameters()
        self.model = KinematicBicycleModel(self.vehicle)
        self.map_radius = map_radius

        self._map_api: Optional[AbstractMap] = None
        self._miss_goal: Optional[StateSE2] = None
        self._route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None

    def name(self):
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._miss_goal = initialization.mission_goal
        logger.info(f"miss_goal x:{self._miss_goal.x}, y:{self._miss_goal.y}")

        self._load_route_dicts(initialization)
        gc.collect()

    def _load_route_dicts(self, initialization: PlannerInitialization) -> None:
        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        self._route_roadblock_ids = initialization.route_roadblock_ids
        for roadblock_id in self._route_roadblock_ids:
            logger.info(f"route_roadblock_id: {roadblock_id}")

            block = self._map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                logger.info(f"route_lane_id: {lane.id}")
                self._route_lane_dict[lane.id] = lane

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        logger.info(f"PDM Open Planner: {self.count}")

        # 提取ego当前状态
        ego_state = current_input.history.ego_states[-1]
        logger.info(f"ego_state x:{ego_state.rear_axle.x}, y:{ego_state.rear_axle.y}x")

        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        # start_roadblock = None
        # current_lane = None
        # distance_diff_threshold = 3.0
        # heading_diff_threshold = np.pi / 4
        # distance_diff_thresh = np.inf
        # for roadblock in roadblock_candidate:
        #     for lane in roadblock.interior_edges:
        #         nearest_state = lane.baseline_path.get_nearest_pose_from_position(ego_state.rear_axle.point)
        #         distance_diff = nearest_state.distance_to(ego_state.rear_axle)
        #         heading_diff = nearest_state.heading - ego_state.rear_axle.heading
        #         if abs(distance_diff) < distance_diff_threshold and abs(heading_diff) < heading_diff_threshold:
        #             if abs(distance_diff) < distance_diff_thresh:
        #                 distance_diff_thresh = abs(distance_diff)
        #                 start_roadblock = roadblock
        #                 current_lane = lane
        #
        # logger.info(f"roadblock: {start_roadblock.id}")

        # for state in current_lane.baseline_path.discrete_path:
        #     logger.info(f"lane x: {state.x}  y: {state.y}  heading: {state.heading}")

        # 绘制车道线轨迹点和Ego位置
        lane_x = [state.x for state in current_lane.baseline_path.discrete_path]
        lane_y = [state.y for state in current_lane.baseline_path.discrete_path]
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y

        # plt.figure(figsize=(10, 6))
        # plt.plot(lane_x, lane_y, label="Lane Trajectory", color="blue", marker="o", linestyle="-")
        # plt.scatter(ego_x, ego_y, label="Ego Position", color="red", s=100, zorder=5)
        # plt.title("Lane Trajectory and Ego Position")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # plt.grid(True)

        # # 保存图片
        # file_path = os.path.join('debug', f'lane_and_ego_position_{self.count}.png')
        # plt.savefig(file_path)
        # plt.close()

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
            next_state = self.model.propagate_state(next_state, next_state.dynamic_car_state,
                                                    TimePoint(int(0.25 * 1e6)))
            trajectory.append(next_state)
        return InterpolatedTrajectory(trajectory)

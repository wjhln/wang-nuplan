from typing import List, Type
import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInput,
    PlannerInitialization,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.state_representation import StateSE2, Point2D
from nuplan.planning.script.run_simulation import logger
from planTF_planner.feature.planTF_feature import PlanTFFeature
from planTF_planner.feature.common.route_utils import route_roadblock_correction
from planTF_planner.feature.common.utils import normalize_angle, rotate_round_z_axis


class PlanTFFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 100.0,
        history_horizon: float = 2.0,
        future_horizon: float = 8.0,
        sampling_interval: float = 0.1,
        max_agents: int = 64,
    ):
        super().__init__()

        self.radius = radius

        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.sampling_interval = sampling_interval
        self.history_samples = int(history_horizon / sampling_interval)
        self.future_samples = int(future_horizon / sampling_interval)

        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width

        self.max_agents = max_agents

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
        ]

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return PlanTFFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "planTF_feature"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        return PlanTFFeature()

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        logger.info("get_features_from_scenario")

        # ego
        cur_ego_state = scenario.initial_ego_state
        past_ego_trajectory = scenario.get_ego_past_trajectory(
            iteration=0,
            time_horizon=self.history_horizon,
            num_samples=self.history_samples,
        )
        future_ego_trajectory = scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )
        ego_state_times = (
            list(past_ego_trajectory) + [cur_ego_state] + list(future_ego_trajectory)
        )
        # other agent
        cur_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects_times = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.history_horizon,
                num_samples=self.history_samples,
            )
        ]
        future_tracked_objects_times = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.history_horizon,
                num_samples=self.history_samples,
            )
        ]
        tracked_objects_times = (
            past_tracked_objects_times
            + [cur_tracked_objects]
            + future_tracked_objects_times
        )

        route_roadblocks_ids = scenario.get_route_roadblock_ids()
        map_api = scenario.map_api
        misson_goal = scenario.get_mission_goal()
        traffic_light_status_list = list(
            scenario.get_traffic_light_status_at_iteration(0)
        )

        return self._build_feature(
            cur_idx=self.history_samples,
            ego_state_times=ego_state_times,
            tracked_objects_times=tracked_objects_times,
            route_roadblocks_ids=route_roadblocks_ids,
            map_api=map_api,
            mission_goal=misson_goal,  # type: ignore
            traffic_light_status_list=traffic_light_status_list,
        )

    def _build_feature(
        self,
        cur_idx: int,
        ego_state_times: List[EgoState],
        tracked_objects_times: List[TrackedObjects],
        route_roadblocks_ids: List[str],
        map_api: AbstractMap,
        mission_goal: StateSE2,
        traffic_light_status_list: List[TrafficLightStatusData],
    ):
        current_state = ego_state_times[cur_idx]
        query_xy = current_state.center

        route_roadblocks_ids = route_roadblock_correction(
            current_state, map_api, route_roadblocks_ids
        )

        data = {}
        data["current_state"] = self._get_ego_current_state(
            ego_state_times[cur_idx], ego_state_times[cur_idx - 1]
        )
        ego_feature = self._get_ego_features(ego_state_times)
        agents_feature = self._get_agent_feature(
            query_xy, cur_idx, tracked_objects_times
        )
        return PlanTFFeature()

    def _get_ego_current_state(self, cur_state: EgoState, pre_state: EgoState):
        cur_velocity = cur_state.dynamic_car_state.rear_axle_velocity_2d.x
        steer_angle, yaw_rate = 0, 0
        if cur_velocity > 0.2:  # 速度低时候不计算
            angle_diff = normalize_angle(
                cur_state.rear_axle.heading - pre_state.rear_axle.heading
            )
            yaw_rate = angle_diff / self.sampling_interval
            steer_angle = np.arctan(
                yaw_rate * self.ego_params.wheel_base / abs(cur_velocity)
            )
            steer_angle = np.clip(steer_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

        state = np.zeros(7, dtype=np.float64)
        state[0:2] = cur_state.rear_axle.array
        state[2] = cur_state.rear_axle.heading
        state[3] = cur_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = cur_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state[5] = steer_angle
        state[6] = yaw_rate
        return state

    def _get_ego_features(self, ego_state_times: List[EgoState]):
        T = len(ego_state_times)
        ego_features = {
            "position": np.zeros((T, 2), dtype=np.float64),
            "heading": np.zeros(T, dtype=np.float64),
            "velocity": np.zeros((T, 2), dtype=np.float64),
            "acceleration": np.zeros((T, 2), dtype=np.float64),
            "shape": np.zeros((T, 2), dtype=np.float64),
            "category": np.array(-1, dtype=np.int8),
            "valid_mask": np.ones(T, dtype=np.bool8),
        }
        for t, state in enumerate(ego_state_times):
            ego_features["position"][t] = state.rear_axle.array
            ego_features["heading"][t] = state.rear_axle.heading
            ego_features["velocity"][t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_velocity_2d.array,
                -state.rear_axle.heading,
            )
            ego_features["acceleration"][t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_acceleration_2d.array,
                -state.rear_axle.heading,
            )
        ego_features["shape"][:] = [self.width, self.length]
        ego_features["category"][...] = self.interested_objects_types.index(
            TrackedObjectType.EGO
        )
        return ego_features

    def _get_agent_feature(
        self,
        query_xy: Point2D,
        cur_idx: int,
        tracked_objects_times: List[TrackedObjects],
    ):
        cur_tracked_objects = tracked_objects_times[cur_idx]
        cur_agents = cur_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )
        # agents_features
        N, T = min(len(cur_agents), self.max_agents), len(tracked_objects_times)
        agents_features = {
            "position": np.zeros((N, T, 2), dtype=np.float64),
            "heading": np.zeros((N, T), dtype=np.float64),
            "velocity": np.zeros((N, T, 2), dtype=np.float64),
            "shape": np.zeros((N, T, 2), dtype=np.float64),
            "category": np.array((N,), dtype=np.int8),
            "valid_mask": np.zeros((N, T), dtype=np.bool8),
        }
        # agents_features
        if N == 0:
            return agents_features

        agents_pose = np.array([agent.center.array for agent in cur_agents])
        distance = np.linalg.norm(agents_pose - query_xy.array[None, :], axis=1)
        agents_track_tokens = np.array([agent.track_token for agent in cur_agents])

        agents_track_tokens = agents_track_tokens[
            np.argsort(distance)[: self.max_agents]
        ]  # 过滤掉距离远的，剩余的顺序不变
        agents_track_dict = {
            track_tokens: idx for idx, track_tokens in enumerate(agents_track_tokens)
        }

        for t, tracked_objects in enumerate(tracked_objects_times):
            for agent in tracked_objects.get_tracked_objects_of_types(
                self.interested_objects_types
            ):
                if agent.track_token not in agents_track_tokens:
                    continue
                idx = agents_track_dict[agent.track_token]
                agents_features["position"][idx, t] = agent.center.array
                agents_features["heading"][idx, t] = agent.center.heading
                agents_features["velocity"][idx, t] = agent.velocity.array
                agents_features["shape"][idx, t] = [agent.box.width, agent.box.length]
                agents_features["valid_mask"][idx, t] = True
                if t == cur_idx:
                    agents_features["category"][idx] = (
                        self.interested_objects_types.index(agent.tracked_object_type)
                    )
        return agents_features

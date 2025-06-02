from typing import List, Type
import numpy as np
import shapely
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
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap, PolygonMapObject
from nuplan.common.actor_state.state_representation import StateSE2, Point2D
from nuplan.planning.script.run_simulation import logger
from planTF_planner.feature.planTF_feature import PlanTFFeature
from planTF_planner.feature.common.route_utils import route_roadblock_correction
from planTF_planner.feature.common.utils import (
    normalize_angle,
    rotate_round_z_axis,
    sample_discrete_path,
)


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
        history = current_input.history
        horizon = self.history_samples + 1
        tracked_objects_list = [
            observation.tracked_objects for observation in history.observations
        ]
        traffic_light_status_list = (
            current_input.traffic_light_data
            if current_input.traffic_light_data is not None
            else []
        )
        return self._build_feature(
            cur_idx=-1,
            ego_state_list=history.ego_states[-horizon:],
            tracked_objects_list=tracked_objects_list,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            map_api=initialization.map_api,
            mission_goal=initialization.mission_goal,
            traffic_light_status_list=traffic_light_status_list,
        )

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
            ego_state_list=ego_state_times,
            tracked_objects_list=tracked_objects_times,
            route_roadblocks_ids=route_roadblocks_ids,
            map_api=map_api,
            mission_goal=misson_goal,  # type: ignore
            traffic_light_status_list=traffic_light_status_list,
        )

    def _build_feature(
        self,
        cur_idx: int,
        ego_state_list: List[EgoState],
        tracked_objects_list: List[TrackedObjects],
        route_roadblocks_ids: List[str],
        map_api: AbstractMap,
        mission_goal: StateSE2,
        traffic_light_status_list: List[TrafficLightStatusData],
    ):
        current_state = ego_state_list[cur_idx]
        query_xy = current_state.center

        route_roadblocks_ids = route_roadblock_correction(
            current_state, map_api, route_roadblocks_ids
        )

        feature = {}
        feature["current_state"] = self._get_ego_current_state(
            ego_state_list[cur_idx], ego_state_list[cur_idx - 1]
        )
        ego_feature = self._get_ego_features(ego_state_list)
        agents_feature = self._get_agent_feature(
            query_xy, cur_idx, tracked_objects_list
        )
        feature["agent"] = {}
        for k in agents_feature:
            feature["agent"][k] = np.concatenate(
                [ego_feature[k][None:...], agents_feature[k]], axis=0
            )
        feature["map"] = self._get_map_feature(
            map_api,
            query_xy,
            route_roadblocks_ids,
            traffic_light_status_list,
            self.radiuss,
        )
        return PlanTFFeature(feature)

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

    def _get_map_feature(
        self,
        map_api: AbstractMap,
        query_xy: Point2D,
        route_roadblocks_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        radius: float,
        sample_points: int = 20,
    ):

        map_objects = map_api.get_proximal_map_objects(
            query_xy, radius, self.polygon_types
        )
        lane_objects = map_objects[SemanticMapLayer.LANE] + [
            SemanticMapLayer.LANE_CONNECTOR
        ]
        crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]

        object_ids = [int(obj.id) for obj in lane_objects + crosswalk_objects]
        object_types = (
            [SemanticMapLayer.LANE] * len(map_objects[SemanticMapLayer.LANE])
            + [SemanticMapLayer.LANE_CONNECTOR]
            * len(map_objects[SemanticMapLayer.LANE])
            + [SemanticMapLayer.LANE] * len(map_objects[SemanticMapLayer.LANE])
        )

        M, P = len(lane_objects) + len(crosswalk_objects), sample_points
        map_feature = {
            "point_position": np.zeros((M, 3, P, 2), dtype=np.float64),
            "point_vector": np.zeros((M, 3, P, 2), dtype=np.float64),
            "point_orientation": np.zeros((M, 3, P), dtype=np.float64),
            "point_side": np.zeros((M, 3), dtype=np.int8),
            "polygon_centor": np.zeros((M, 3), dtype=np.float64),
            "polygon_position": np.zeros((M, 2), dtype=np.float64),
            "polygon_orientation": np.zeros(M, dtype=np.float64),
            "polygon_type": np.zeros(M, dtype=np.int8),
            "polygon_on_route": np.zeros(M, dtype=np.bool8),
            "polygon_tl_status": np.zeros(M, dtype=np.int8),
            "polygon_has_speed_limit": np.zeros(M, dtype=np.bool8),
            "polygon_speed_limit": np.zeros(M, dtype=np.float64),
        }

        for lane in lane_objects:
            center_line = sample_discrete_path(  # TODO:待学习：路径插值
                lane.baseline_path.discrete_path, sample_points + 1
            )
            left_lane = sample_discrete_path(
                lane.left_boundary.discrete_path, sample_points + 1
            )
            right_lane = sample_discrete_path(
                lane.right_boundary.discrete_path, sample_points + 1
            )
            edges = np.stack([center_line, left_lane, right_lane], axis=0)

            object_id = int(lane.id)
            idx = object_ids.index(object_id)
            map_feature["point_position"][idx] = edges[:, :-1]
            map_feature["point_vector"][idx] = edges[:, 1:] - edges[:, :-1]
            map_feature["point_orientation"][idx] = np.arctan2(
                map_feature["point_vector"][idx, :, :, 1],
                map_feature["point_vector"][idx, :, :, 0],
            )
            map_feature["point_side"][idx] = np.arange(3)

            map_feature["polygon_centor"][idx] = np.concatenate(
                [
                    center_line[int(sample_points / 2)],
                    [map_feature["point_orientation"][idx, 0, int(sample_points / 2)]],
                ],
                axis=-1,
            )
            map_feature["polygon_position"][idx] = center_line[0]
            map_feature["polygon_orientation"][idx] = map_feature["point_orientation"][
                idx, 0, 0
            ]
            map_feature["polygon_type"][idx] = self.polygon_types.index(
                object_types[idx]
            )

            route_ids = set(int(route_id) for route_id in route_roadblocks_ids)
            map_feature["polygon_on_route"][idx] = (
                int(lane.get_roadblock_id()) in route_ids
            )

            traffic_light_status_dict = {
                traffic_light_state.lane_connector_id: traffic_light_state.status
                for traffic_light_state in traffic_light_status
            }
            map_feature["polygon_tl_status"][idx] = int(
                traffic_light_status_dict[object_id]
                if object_id in traffic_light_status_dict
                else TrafficLightStatusType.UNKNOWN
            )
            map_feature["polygon_speed_limit"][idx] = (
                lane.speed_limit_mps if lane.speed_limit_mps else 0
            )
            map_feature["polygon_has_speed_limit"][idx] = (
                lane.speed_limit_mps is not None
            )

        for crosswalk in crosswalk_objects:
            idx = object_ids.index(int(crosswalk.id))
            edges = self._get_crosswalk_edges(crosswalk)
            map_feature["point_position"][idx] = edges[:, :-1]
            map_feature["point_vector"][idx] = edges[:, 1:] - edges[:, :-1]
            map_feature["point_orientation"][idx] = np.arctan2(
                map_feature["point_vector"][idx, :, :, 1],
                map_feature["point_vector"][idx, :, :, 0],
            )
            map_feature["point_side"][idx] = np.arange(3)

            map_feature["polygon_centor"][idx] = np.concatenate(
                [
                    edges[0, int(sample_points / 2)],
                    [map_feature["point_orientation"][idx, 0, sample_points / 2]],
                ]
            )
            map_feature["polygon_position"][idx] = edges[0, 0]
            map_feature["polygon_orientation"][idx] = map_feature["point_orientation"][
                idx, 0, 0
            ]
            map_feature["polygon_type"][idx] = self.polygon_types.index(
                object_types[idx]
            )
            map_feature["polygon_on_route"][idx] = False
            map_feature["polygon_tl_status"][idx] = TrafficLightStatusType.UNKNOWN
            map_feature["polygon_has_speed_limit"][idx] = False
            map_feature["polygon_speed_limit"][idx] = 0

        return map_feature

    def _get_crosswalk_edges(self, crosswalk: PolygonMapObject, sample_nums: int = 21):
        # TODO:待学习：最小旋转矩形
        bbox = shapely.minimum_rotated_rectangle(crosswalk.polygon)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge_r = coords[[3, 0]]  # [2 ,2]
        edge_l = coords[[2, 1]]
        edge_m = (edge_r + edge_l) * 0.5
        edges = np.stack([edge_m, edge_l, edge_r], axis=0)  # [3, 2, 2]
        vector = edges[:, 1] - edges[:, 0]  # [3, 2]
        step = np.linspace(0, 1, sample_nums, endpoint=True)[None, :]
        # [1, sample_nums]
        points = edges[:, 0, None, :] + vector[:, None, :] * step[:, :, None]
        # [3, sample_nums, 2]
        return points

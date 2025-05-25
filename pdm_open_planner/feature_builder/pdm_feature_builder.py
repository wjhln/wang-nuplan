from typing import Tuple, Type, List, Optional
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInput,
    PlannerInitialization,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from pdm_open_planner.feature.pdm_feature import PDMFeature
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType,
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_yaw_rate,
    extract_ego_acceleration,
)
from pdm_open_planner.model.pdm_open_model import PDMOpenModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


def _get_planner_params_from_scenario(
    scenario: AbstractScenario,
) -> Tuple[PlannerInput, PlannerInitialization]:
    print(scenario.database_interval)

    batch_size = int(2 / scenario.database_interval + 1)

    planner_initialization = PlannerInitialization(
        scenario.get_route_roadblock_ids(),
        scenario.get_mission_goal(),
        scenario.map_api,
    )

    history = SimulationHistoryBuffer.initialize_from_scenario(
        batch_size, scenario, DetectionsTracks
    )

    planner_input = PlannerInput(
        SimulationIteration(index=0, time_point=scenario.start_time),
        history,
        list(scenario.get_traffic_light_status_at_iteration(0)),
    )

    return planner_input, planner_initialization


class PDMOpenFeatureBuilder(AbstractFeatureBuilder):

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
    ):
        self.trajectory_sampling = trajectory_sampling
        self.history_sampling = history_sampling
        self.centerline_samples = centerline_samples
        self.centerline_interval = centerline_interval

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "pdm_open_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return PDMFeature

    def get_features_from_scenario(self, scenario: AbstractScenario) -> PDMFeature:
        ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                0, self.history_sampling.time_horizon, self.history_sampling.num_poses
            )
        ] + [scenario.initial_ego_state]

        curren_input, initialtion = _get_planner_params_from_scenario(scenario)

        return 

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PDMFeature:
        ego_states = current_input.history.ego_states

        # 历史轨迹提取
        ego_position = _get_ego_position(ego_states)
        ego_velocity = _get_ego_velocity(ego_states)
        ego_acceleration = _get_ego_acceleration(ego_states)

        return PDMFeature(
            ego_position,
            ego_velocity,
            ego_acceleration,
            planner_centerline,
            planner_trajectory,
        )


def _get_ego_position(ego_states: List[EgoState]) -> FeatureDataType:
    return build_ego_features(ego_states)


def _get_ego_velocity(ego_states: List[EgoState]) -> FeatureDataType:
    v_x = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.x for ego_state in ego_states]
    )
    v_y = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.y for ego_state in ego_states]
    )
    v_yaw = extract_ego_yaw_rate(ego_states)
    return np.stack([v_x, v_y, v_yaw], axis=-1)


def _get_ego_acceleration(ego_states: List[EgoState]) -> FeatureDataType:
    a_x = extract_ego_acceleration(ego_states, "x")
    a_y = extract_ego_acceleration(ego_states, "y")
    a_yaw = extract_ego_yaw_rate(ego_states, 2, 3)
    return np.stack([a_x, a_y, a_yaw], axis=-1)


def create_pdm_feature(
    model: PDMOpenModel,
    planner_input: PlannerInput,
    centerline: PDMPath,
    closed_loop_trajectory: Optional[InterpolatedTrajectory] = None,
    device: str = "cpu",
) -> PDMFeature:
    history = planner_input.history

    current_ego_state = history.ego_states[-1]
    past_ego_states = history.ego_states[:-1]
    indices = sample_indices_with_time_horizon(
        model.history_sampling.num_poses,
        model.history_sampling.time_horizon,
        history.sample_interval,
    )
    sample_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
    sample_past_ego_states = sample_past_ego_states + [current_ego_state]

    ego_position = _get_ego_position(sample_past_ego_states)
    ego_velocity = _get_ego_velocity(sample_past_ego_states)
    ego_acceleration = _get_ego_acceleration(sample_past_ego_states)

    return PDMFeature(ego_position, ego_velocity, ego_acceleration)

from typing import Type
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
from pdm_open_planner.feature.pdm_feature import PDMFeature
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features

class PDMOpenFeatureBuilder(AbstractFeatureBuilder):
    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "pdm_open_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return PDMFeature

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        # ego_states = current_input.history.ego_states

        # ego_position = build_ego_features(ego_states)
        # ego_velocity = 
        # return PDMFeature(
        #     ego_position,
        #     ego_velocity,
        #     ego_acceleration,
        #     planner_centerline,
        #     planner_trajectory,
        # )
        pass

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        pass

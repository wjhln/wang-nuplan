from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature

from planTF_planner.feature.planTF_feature import PlanTFFeature


class PlanTFFeatureBuilder(AbstractFeatureBuilder):
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return PlanTFFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "planTF_feature"

    def get_features_from_simulation(self, current_input: PlannerInput,
                                     initialization: PlannerInitialization) -> AbstractModelFeature:
        return PlanTFFeature()

    def get_features_from_scenario(self, scenario: AbstractScenario) -> AbstractModelFeature:
        return PlanTFFeature()
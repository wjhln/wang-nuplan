from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import  AbstractTargetBuilder
from pdm_open_planner.feature.pdm_feature import PDMFeature

class PDMOpenTargetBuilder(AbstractTargetBuilder):
    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "trajectory"

    def get_targets(self, scenario: AbstractScenario) -> AbstractModelFeature:
        return PDMFeature(0,0,0,0,0)

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return PDMFeature
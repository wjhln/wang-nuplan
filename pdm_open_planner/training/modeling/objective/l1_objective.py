from typing import List
import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import (
    AbstractObjective,
)
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)


class L1Objective(AbstractObjective):
    def __init__(self):
        pass

    def name(self) -> str:
        return super().name()

    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return ["trajectory"]

    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        scenarios: ScenarioListType,
    ) -> torch.Tensor:

        return torch.tensor(0)

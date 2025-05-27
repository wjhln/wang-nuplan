from typing import List, Dict, Any

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature

class PlanTFFeature(AbstractModelFeature):
    date: Dict[str, Any] = None

    def to_feature_tensor(self) -> AbstractModelFeature:
        pass

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        pass

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        pass

    def unpack(self) -> List[AbstractModelFeature]:
        pass
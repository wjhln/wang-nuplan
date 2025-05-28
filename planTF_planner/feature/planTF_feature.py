from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature

@dataclass
class PlanTFFeature(AbstractModelFeature):
    date: Dict[str, Any] = None

    @classmethod
    def collate(cls, batch: List[PlanTFFeature]) -> PlanTFFeature:
        batch_data = {}
        for key in ["agent", "map"]:
            batch_data[key] = {

            }

        return PlanTFFeature()

    def to_feature_tensor(self) -> PlanTFFeature:
        return PlanTFFeature()

    def to_device(self, device: torch.device) -> PlanTFFeature:
        return PlanTFFeature()

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlanTFFeature:
        return PlanTFFeature()

    def unpack(self) -> List[PlanTFFeature]:
        return [PlanTFFeature()]
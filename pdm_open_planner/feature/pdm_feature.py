from typing import List, Dict, Any

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType,
    AbstractModelFeature,
    to_tensor,
)
from dataclasses import dataclass


@dataclass
class PDMFeature(AbstractModelFeature):
    ego_position: FeatureDataType
    ego_velocity: FeatureDataType
    ego_acceleration: FeatureDataType
    planner_centerline: FeatureDataType
    planner_trajectory: FeatureDataType


    def to_feature_tensor(self) -> AbstractModelFeature:
        return PDMFeature(
            to_tensor(self.ego_position),
            to_tensor(self.ego_velocity),
            to_tensor(self.ego_acceleration),
            to_tensor(self.planner_centerline),
            to_tensor(self.planner_trajectory),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return PDMFeature(
            to_tensor(self.ego_position).to(device),
            to_tensor(self.ego_velocity).to(device),
            to_tensor(self.ego_acceleration).to(device),
            to_tensor(self.planner_centerline).to(device),
            to_tensor(self.planner_trajectory).to(device),
        )

    def unpack(self) -> List[AbstractModelFeature]:
        return [
            PDMFeature(
                ego_position[None],
                ego_velocity[None],
                ego_acceleration[None],
                planner_centerline[None],
                planner_trajectory[None],
            )
            for ego_position, ego_velocity, ego_acceleration, planner_centerline, planner_trajectory in zip(
                self.ego_position,
                self.ego_velocity,
                self.ego_acceleration,
                self.planner_centerline,
                self.planner_trajectory,
            )
        ]

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return PDMFeature(
            data["ego_position"],
            data["ego_velocity"],
            data["ego_acceleration"],
            data["planner_centerline"],
            data["planner_trajectory"],
        )
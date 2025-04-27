from dataclasses import dataclass
from typing import List, Dict, Any

import torch

from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, \
    FeatureDataType, to_tensor


@dataclass
class PDMFeature(AbstractModelFeature):
    ego_states: FeatureDataType
    ego_velocity: FeatureDataType
    ego_acceleration: FeatureDataType
    centerline: FeatureDataType
    trajectory: FeatureDataType

    def unpack(self) -> List[AbstractModelFeature]:
        return [
            PDMFeature(
                ego_states=self.ego_states[i],
                ego_velocity=self.ego_velocity[i],
                ego_acceleration=self.ego_acceleration[i],
                centerline=self.centerline[i],
                trajectory=self.trajectory[i],
            )
            for i in range(len(self.ego_states))
        ]

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return PDMFeature(
            ego_states=data["ego_states"],
            ego_velocity=data["ego_velocity"],
            ego_acceleration=data["ego_acceleration"],
            centerline=data["centerline"],
            trajectory=data["trajectory"],
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return PDMFeature(
            ego_states=self.ego_states.to(device),
            ego_velocity=self.ego_velocity.to(device),
            ego_acceleration=self.ego_acceleration.to(device),
            centerline=self.centerline.to(device),
            trajectory=self.trajectory.to(device),
        )


    def to_feature_tensor(self) -> AbstractModelFeature:
        return PDMFeature(
            ego_states=to_tensor(self.ego_states),
            ego_velocity=to_tensor(self.ego_velocity),
            ego_acceleration=to_tensor(self.ego_acceleration),
            centerline=to_tensor(self.centerline),
            trajectory=to_tensor(self.trajectory),
        )


def create_pdm_feature(planner_input: PlannerInput) -> PDMFeature:
    indices = sample_indices_with_time_horizon(planner_input.history.)
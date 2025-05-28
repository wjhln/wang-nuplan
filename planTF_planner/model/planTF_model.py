import logging

import torch
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder
from torch import nn

from planTF_planner.feature.planTF_feature import PlanTFFeature
from planTF_planner.feature.planTF_feature_builder import PlanTFFeatureBuilder
from nuplan.planning.script.run_simulation import logger

from planTF_planner.model.mlp import build_mlp

trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8)


class PlanTFModel(TorchModuleWrapper):

    def __init__(
            self,
            dim = 128,
    ):
        feature_builder = PlanTFFeatureBuilder()
        target_builder = EgoTrajectoryTargetBuilder(trajectory_sampling)
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[target_builder],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.pos_embedding = build_mlp(4, [dim] * 2) # [x, y, cos(θ), sin(θ)]

    def forward(self, features: FeaturesType) -> TargetsType:
        logger.warning("aaaaaaaaaaaaaaaaa")
        # self.pos_embedding(features)
        pass

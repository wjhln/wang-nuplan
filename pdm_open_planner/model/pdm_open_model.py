from torch import nn
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pdm_open_planner.feature_builder.feature_builder import PDMOpenFeatureBuilder


class PDMOpenModel(TorchModuleWrapper):
    def __init__(self):
        trajectory_sampling = TrajectorySampling(num_poses=16, time_horizon=0.5)
        feature_builders = [PDMOpenFeatureBuilder()]
        super().__init__()

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        :return: The results of the inference as a TargetsType.
        """
        pass

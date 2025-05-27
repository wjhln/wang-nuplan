from typing import Type

from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput, PlannerInitialization
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class PlanTF(AbstractPlanner):
    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        pass

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        pass


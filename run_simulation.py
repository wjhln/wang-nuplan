import hydra
from loguru import logger
from nuplan.planning.script.run_simulation import run_simulation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from pdm_planner.pdm_open_planner import PDMOpenPlanner

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path='nuplan/planning/script/config/simulation')
cfg = hydra.compose(config_name='default_simulation', overrides=[
    'job_name=wang',
    'output_dir=./exp',
    'ego_controller=perfect_tracking_controller',
    'observation=idm_agents_observation',
    '+simulation=closed_loop_reactive_agents',
    'scenario_filter.limit_total_scenarios=1',
    'worker=sequential',
    'metric_aggregator=closed_loop_reactive_agents_weighted_average',
])

planner1 = PDMOpenPlanner()
planner2 = SimplePlanner(horizon_seconds=10.0, sampling_time=0.25, acceleration=[0.0, 0.0], max_velocity = 0)
planner = [planner1, planner2]
run_simulation(cfg, planner)
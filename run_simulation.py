import hydra

from nuplan.planning.script.run_simulation import run_simulation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from pdm_open_planner.simulation.pdm_open_planner import PDMOpenPlanner

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path='config')
cfg = hydra.compose(config_name='default_simulation', overrides=[
    'output_dir=./exp/closed_loop_reactive_agents',
    'ego_controller=perfect_tracking_controller',
    'observation=idm_agents_observation',
    '+simulation=closed_loop_reactive_agents',
    'scenario_builder=nuplan_mini',
    'scenario_filter=all_scenarios',
    # 'scenario_filter.scenario_types=[starting_left_turn]',
    # 'scenario_filter.limit_total_scenarios=1',
    'scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"]',
    'worker=sequential'
])

planner1 = PDMOpenPlanner()
planner2 = SimplePlanner(horizon_seconds=10.0, sampling_time=0.25, acceleration= [0.2, 0.0], max_velocity = 2)
planner = [planner1]
run_simulation(cfg, planner)
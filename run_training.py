import hydra

from nuplan.planning.script.run_training import main as run_training

hydra.core.global_hydra.GlobalHydra.instance().clear() # type: ignore
hydra.initialize(config_path='config')
cfg = hydra.compose(config_name='default_training', overrides=[
    # 'scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"]',
    # 'scenario_builder=nuplan_mini',
    # 'scenario_filter=one_continuous_log',
    # 'scenario_filter.limit_total_scenarios=1',

    # '+training=training_pdm_open_model',
    # 'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://pdm_open_planner.planning.script.config.common"]'
])

run_training(cfg)  
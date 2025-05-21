import hydra

from nuplan.planning.script.run_training import main as run_training

hydra.core.global_hydra.GlobalHydra.instance().clear() # type: ignore
hydra.initialize(config_path='config')
cfg = hydra.compose(config_name='default_training', overrides=[
    # '+training=training_pdm_open_model',
    # 'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://pdm_open_planner.planning.script.config.common"]'
])

run_training(cfg)
import hydra

from nuplan.planning.script.run_training import main as run_training

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path='config')
cfg = hydra.compose(config_name='default_training', overrides=[
    '+training=training_pdm_open_model',
])

run_training(cfg)
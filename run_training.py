import hydra

from nuplan.planning.script.run_training import main as run_training

hydra.core.global_hydra.GlobalHydra.instance().clear() # type: ignore
hydra.initialize(config_path='config')
cfg = hydra.compose(config_name='train_planTF', overrides=[
    # 'scenario_filter.scenario_tokens=["c742dfbe4e4c5b60"]',
    # 'scenario_builder=nuplan_mini',
    # 'scenario_filter=all_scenarios',
    # 'scenario_filter.limit_total_scenarios=1',

    # '+training=training_pdm_open_model',
    # 'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://pdm_open_planner.planning.script.config.common"]'
])

run_training(cfg)

# python run_training.py \
#   py_func=train +training=train_planTF \
#   worker=single_machine_thread_pool worker.max_workers=32 \
#   scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_plantf_1M cache.use_cache_without_dataset=true \
#   data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
#   lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
#   lightning.trainer.params.val_check_interval=0.5 \
#   wandb.mode=online wandb.project=nuplan wandb.name=plantf
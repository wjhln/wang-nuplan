import os
import hydra
import hydra.core
from pathlib import Path
from nuplan.planning.script.run_nuboard import main as main_nuboard


hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path='nuplan/planning/script/config/nuboard')

exp_dir = '/home/wang/Project/wang-nuplan/exp'
nuboard_files = list(Path(exp_dir).glob('**/*.nuboard'))

if nuboard_files:
    # 按文件修改时间排序，获取最新的.nuboard文件
    latest_file = max(nuboard_files, key=lambda x: x.stat().st_mtime)
    experiment_path = str(latest_file)
    print(f"使用最新的.nuboard文件: {experiment_path}")
    cfg = hydra.compose(config_name='default_nuboard', overrides=[
        'scenario_builder=nuplan_mini',
        f'simulation_path={experiment_path}',
    ])
    main_nuboard(cfg)
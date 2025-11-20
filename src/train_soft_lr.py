from pathlib import Path
import yaml
import wandb
import torch
from typing import NamedTuple

from utils import log_yaml, get_latest_checkpoint_path_from_dir
from parse import process_config_from_file
from train import TrainerSteps

def perform_training(conf_path: Path, include_path: Path, checkpoint_dir: Path | None = None, 
                     resume_training: bool = True, wandb_mode: str = "online") -> Path:
    
    processed_config, parsed_config = process_config_from_file(filename=str(conf_path), 
                                                               include=str(include_path),
                                                               checkpoint_path=str(get_latest_checkpoint_path_from_dir(checkpoint_dir)) if checkpoint_dir is not None else None,
                                                               resume_training=resume_training)
    
    wandb.init(mode=wandb_mode, config=parsed_config)
    log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
    
    trainer = TrainerSteps(**processed_config)
    trainer.train()
    output_dir = trainer.get_output_dir()
    wandb.finish()

    return output_dir

class TrainInfo(NamedTuple):
    conf_path: Path
    output_dir_name: str

# Set constants, variables and paths
WANDB_MODE = "online"
CONF_DIR = Path(__file__).parent.parent / "conf" / "train" / "poly_playground_paper" / "soft_prompt_learning_rates"
CONF_INCLUDE_DIR = Path(__file__).parent.parent / "conf" / "include" # Yaml-conf files to include

base_model_dir = Path(__file__).parent.parent / "models" / "poly_base_model"

fine_tune_train_info = [TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_2_fixed_0_lr_0_05.yml", output_dir_name="poly_soft_prompting_2_model_fixed_0_lr_0_05"),
                        TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_2_fixed_1_lr_0_05.yml", output_dir_name="poly_soft_prompting_2_model_fixed_1_lr_0_05"),
                        TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_2_fixed_3_lr_0_05.yml", output_dir_name="poly_soft_prompting_2_model_fixed_3_lr_0_05"),
                        TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_2_fixed_5_lr_0_05.yml", output_dir_name="poly_soft_prompting_2_model_fixed_5_lr_0_05")]
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_1_lr_0_05.yml", output_dir_name="poly_soft_prompting_50_model_fixed_1_lr_0_05"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_3_lr_0_05.yml", output_dir_name="poly_soft_prompting_50_model_fixed_3_lr_0_05"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_5_lr_0_05.yml", output_dir_name="poly_soft_prompting_50_model_fixed_5_lr_0_05")]
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_005.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_005"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_05.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_05"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_5.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_5")]
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_0005.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_0005"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_000005.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_000005"),
                        #TrainInfo(conf_path=CONF_DIR / "train_chebyshev_soft_prompting_50_fixed_0_lr_0_0000005.yml", output_dir_name="poly_soft_prompting_50_model_fixed_0_lr_0_0000005")]

# Check cuda status
print("CUDA available:", torch.cuda.is_available()) 

# 0. Save results path
results: list[tuple[Path, TrainInfo]] = []  # List of tuples (output_dir, train_info)

# 2. Train finetuning methods
results.extend([(perform_training(conf_path=train_info.conf_path, include_path=CONF_INCLUDE_DIR, 
                                  checkpoint_dir=base_model_dir, wandb_mode=WANDB_MODE, resume_training=False),
                 train_info) for train_info in fine_tune_train_info])

# 4. Rename output directories
[output_dir.rename((output_dir.parent / output_dir_name).absolute()) for output_dir, (_, output_dir_name) in results]
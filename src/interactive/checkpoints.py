
from core import ContextModel
from parse import process_config_from_file

def load_checkpoint_with_config(config_file: str, checkpoint_file: str, import_configs: bool = True) -> ContextModel:
    trainer, _ = process_config_from_file(
        filename=config_file,
        include="../conf/include" if import_configs else None,
        checkpoint_path=checkpoint_file
    )
    return trainer["model"]

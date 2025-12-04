from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm","gpt2_var_glu","gpt2_var_moe"])),
    "glu_type": merge(tstring, allowed(["glu", "geglu", "swiglu","None"])),
    "num_experts":merge(tinteger, required),
    "vocab_size":merge(tinteger, required),
    "top_k":merge(tinteger, required),    
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "degree": stdict(curriculum_base_schema), #modify as degree
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "polynomial_regression", #add polynomial_regression here
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian","uniform"])), #add uniform here
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}

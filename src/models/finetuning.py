from peft import LoraConfig, get_peft_model # pyright: ignore[reportPrivateImportUsage]
from typing import Any
import torch

from .transformer import GPT2


class FineTuningGPT2Model(GPT2):

    def _load_weights_if_matching(self, model_weights: dict[str, Any] | None) -> bool:
        
        if model_weights is not None and model_weights.keys() == self.state_dict().keys():
            self.load_state_dict(model_weights, strict=True)
            return True
        
        return False
        

class LoraGPT2Model(FineTuningGPT2Model):

    def __init__(self, base_model_specs: dict[str, Any], lora_config: dict[str, Any], **kwargs: dict[Any, Any]):

        model_weights = kwargs.pop('model_weights', None)

        super(LoraGPT2Model, self).__init__(**(base_model_specs | kwargs))

        # If given weights are just for base model, load them
        weights_loaded = self._load_weights_if_matching(model_weights)

        # Freeze original model
        for param in self.parameters():
            param.requires_grad = False

        self._backbone = get_peft_model(self._backbone, LoraConfig(**lora_config)) # pyright: ignore[reportArgumentType]

        # If given weights are for the whole model, load them
        weights_loaded |= self._load_weights_if_matching(model_weights)

        if not weights_loaded:
            raise ValueError("Could not load weights")

        print("LoraGPT2Model trainable parameters:", self.get_number_of_trainable_parameters())

class SoftPromptingGPT2Model(FineTuningGPT2Model):
        
    def __init__(self, base_model_specs: dict[str, Any], prompt_pairs: int, **kwargs: dict[str, Any]):

        model_weights = kwargs.pop('model_weights', None)

        super(SoftPromptingGPT2Model, self).__init__(**(base_model_specs | kwargs))

        # If given weights are just for base model, load them
        weights_loaded = self._load_weights_if_matching(model_weights)

        # Freeze original model
        for param in self.parameters():
            param.requires_grad = False

        self.prompt_pairs = prompt_pairs

        # Prepare soft prompt
        start_inputs = torch.randn((1, 2*self.prompt_pairs, 1))
        self.soft_prompt = torch.nn.Parameter(self._read_in(start_inputs))

        # If given weights are for the whole model, load them
        weights_loaded |= self._load_weights_if_matching(model_weights)

        if not weights_loaded:
            raise ValueError("Could not load weights")

        print("SoftPromptingGPT2Model trainable parameters:", self.get_number_of_trainable_parameters())

    def forward(self, xs: torch.Tensor, ys: torch.Tensor):

        self.to(xs.device) # pyright: ignore[reportArgumentType,reportAttributeAccessIssue]

        zs = self.interleave(xs, ys)

        embeds = self._read_in(zs)
        soft_prompts = self.soft_prompt.repeat((xs.shape[0], 1, 1))
        embeds = torch.cat((soft_prompts, embeds), dim=1)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state # pyright: ignore[reportCallIssue]
        prediction = self._read_out(output)

        return prediction[:, 2*self.prompt_pairs::2] # predict only on xs

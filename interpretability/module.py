from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from .utils import HookPoint
from .gpt.gpt import GPT
import numpy as np
import os
import torch.nn.functional as F
from typing import Literal

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelData:
    activations: dict = field(default_factory=lambda: dict())
    gradients: dict = field(default_factory=lambda: dict())
    parameters: dict = field(default_factory=lambda: dict())
    inputs = None
    targets = None
    outputs = None

def _reduce(values: list, reduction: Literal[None, "mean", "max"], dim: int):
    if reduction is None:
        return values
    elif reduction == "mean":
        return [v.mean(dim=dim) for v in values]
    elif reduction == "max":
        return [v.max(dim=dim) for v in values]
    else:
        raise ValueError("Reduction operation not recognised")


class BaseModule(ABC):
    def __init__(self, model: torch.nn.Module):
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.data = ModelData()
        self.hooks = []
        for n, p in self.model.named_parameters():
            self.data.parameters[n] = p

    def initialise_dataloader(self, **kwargs):
        self.dataloader = self.custom_dataloader(**kwargs)

    def forward(self, inputs: torch.tensor = None, targets: torch.tensor = None):
        self.model.zero_grad()
        self.register_hooks()
        if inputs is None:
            self.step += 1
            self.data.inputs, self.data.targets = next(self.dataloader)
        else:
            self.data.inputs, self.data.targets = inputs, targets
        self.data.outputs = self.custom_forward(self.data.inputs)
        self.remove_hooks()

    def backward(self, targets: torch.tensor = None):
        self.data.targets = targets if targets is not None else self.data.targets
        assert self.data.targets is not None, "Cannot complete backward pass, no targets provided."
        loss = self.custom_loss(self.data.outputs, self.data.targets)
        loss.backward()
        for n, p in self.model.named_parameters():
            self.data.gradients[n] = p.grad

    def register_hooks(self):
        for name, module in self.model.named_modules():
            forward_hook = module.register_forward_hook(self._get_hook(name))
            self.hooks.append(forward_hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    self.data.activations[name] = output.detach().cpu()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @abstractmethod
    def custom_dataloader(self, **kwargs) -> tuple:
        """
        Should be overidden inside child class to match specific model.
        Returns: 
            Generator that yields a tuple of (inputs, targets)
        """
        raise NotImplementedError

    @abstractmethod
    def custom_forward(self, inputs, **kwargs):
        """
        Should be overidden inside child class to match specific model.
        Arguments: 
            inputs - value(s) returned by next(custom_dataloader())[0]
        Returns:
            outputs - value(s) outputed by the model
        """
        raise NotImplementedError

    @abstractmethod
    def custom_loss(self, outputs, targets, **kwargs) -> torch.tensor:
        """
        Should be overidden inside child class to match specific model.
        Arguments:
            outputs - value(s) returned by custom_forward()
            targets - value(s) returned by next(custom_dataloader())[1]
        Returns:
            loss - torch tensor representing a single loss value
        """
        raise NotImplementedError

    def get_activation_values_by_tag(self, tag: str, reduction: Literal[None, "mean", "max"] = "mean", dim: int = 0) -> list:
        """
        Arguments:
            tag - string corresponding to the HookPoint tag inside the model
            reduction - how to reduce the activations (usually along the batch dimension)
            dim - dimension along which to reduce (usually the batch dimension)
        Returns:
            list of activations for all the modules matching the given tag
        """
        names = []
        for name, module in self.model.named_modules():
            if isinstance(module, HookPoint) and tag in module.tags:
                names.append(name)
        values = [self.data.activations[name] for name in names]
        return _reduce(values, reduction, dim)

    def get_activation_value_by_name(self, name: str, reduction: Literal[None, "mean", "max"] = "mean", dim: int = 0) -> torch.Tensor:
        """
        Arguments:
            name - string corresponding to the full module name (matching model.named_modules())
            reduction - how to reduce the activations (usually along the batch dimension)
            dim - dimension along which to reduce (usually the batch dimension)
        Returns:
            activation for a specific module name
        """
        value = [self.data.activations[name]]
        return _reduce(value, reduction, dim)[0]


class GPTModule(BaseModule):
    def __init__(
        self,
        batch_size: int = 1,
        gpt_version: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt-xl"] = "gpt2"
    ):
        model = GPT.from_pretrained(gpt_version)
        super().__init__(model)
        self.initialise_dataloader(bs=batch_size)

    def custom_dataloader(self, bs: int, block_size: int = 100):
        data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'gpt/train.bin'))
        assert os.path.exists(data_path), "No data found. run `python3 gpt/shakespeare.py` to prepare train.bin"
        data = np.memmap(data_path, dtype=np.uint16, mode='r')

        def get_batch():
            ix = torch.randint(len(data) - block_size, (bs,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            return (x, y)
        
        def generator():
            while True:
                yield get_batch()
        
        return generator()

    def custom_forward(self, inputs: torch.tensor) -> dict:
        return self.model(inputs)

    def custom_loss(self, outputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

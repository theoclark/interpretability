"""
Creates a nice interface for easily analysing activations within a model.
Each new type of model will need its own child class of `BaseInterpretabilityModule`
in order to define custom dataloading and forward passes. This is designed to be
used inside a notebook.

Activations are stored inside `BaseInterpretabilityModule.data.activations`.
Parameters can also be accessed in this way, as can the input, output and target
of the model.

Two methods of targeting activations are supported:

1) HookPoint tags. Placing hook points within the model definition enable specific
activations to be targeted. A custom tag should be passed so that they can be easily accessed later.
Good for targeting specific points but requires editing the model definition. Example usage shown below.

2) Module names. Doesn't require editing the model definition but more limited in terms of which activations
you can obtain. All activations, including those from hook points, can be accessed by going into:
`interp.data.activations` or by calling `interp.get_activation_by_name("module_name")`.

-----------------------------------------------------------------------------------------------------

Example Usage:

The following lines of code will generate a diagram of an attention head.

Inside Model definition:

from utils import HookPoint

# In init method of MultiHeadAttention:
self.attention_hook = HookPoint(tags=["attention"])

# In forward method:
x = self.attention_hook(x)    <--- add identity function where you want to capture the activation

Inside Notebook:

```
interp = ModelInterpretabilityModule(model)
interp.forward()            # completes one forward pass
attn = interp.get_activation_values_by_tag("attention")
show_attention_head(attn, layer=0, head=0)
```

-----------------------------------------------------------------------------------------------------

Utility functions for generating various plots of activations can be found in utils.py

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from utils import HookPoint
from gpt import GPT
import numpy as np
import os
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelData:
    activations: dict = field(default_factory=lambda: dict())
    gradients: dict = field(default_factory=lambda: dict())
    external: dict = field(default_factory=lambda: dict())
    parameters: dict = field(default_factory=lambda: dict())


class BaseInterpretabilityModule(ABC):
    def __init__(self, model: torch.nn.Module):
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.data = ModelData()
        self.hooks = []
        for n, p in self.model.named_parameters():
            self.data.parameters[n] = p

    def initialise(self, **kwargs):
        self.dataloader = self.custom_dataloader(**kwargs)

    def forward(self):
        self.model.zero_grad()
        self.step += 1
        self.register_hooks()
        self.data.external = self.custom_forward(self.dataloader, self.model)
        self.remove_hooks()

    def backward(self):
        out = self.data.external["out"]
        target = self.data.external["target"]
        loss = self.custom_loss(out, target)
        loss.backward()
        for n, p in self.model.named_parameters():
            self.data.gradients[n] = p.grad

    def register_hooks(self):
        for name, module in self.model.named_modules():
            forward_hook = module.register_forward_hook(self._get_hook(name))
            self.hooks.append(forward_hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            output_ = output[0].detach().cpu()
            with torch.no_grad():
                self.data.activations[name] = output_

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @abstractmethod
    def custom_dataloader(self, data_path: str):
        """
        Should be overidden inside child class to match specific model.
        Should  be used to generate data samples (and possibly target samples)
        that are accessed in `custom_forward`. `self.step` could be used with
        map-style datasets.
        """
        raise NotImplementedError

    @abstractmethod
    def custom_forward(self, dataloader, model: torch.nn.Module) -> dict:
        """
        Should be overidden inside child class to match specific model.
        Should return all inputs and outputs of the model in a dictionary with
        appropriate keys {"data", "out", "target"}. The dataloader is the same
        as the one returned in `custom_dataloader`. These values can then be
        accessed as attributes in `InterpetabilityModule.data.external`.
        """
        raise NotImplementedError

    @abstractmethod
    def custom_loss(self, out: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        Should be overidden inside child class to match specific model.
        Should return a tensor representing a single loss value.
        """
        raise NotImplementedError

    def get_activation_values_by_tag(self, tag: str) -> list:
        """
        returns a list of activations for all the modules
        matching the given tag
        """
        names = []
        for name, module in self.model.named_modules():
            if isinstance(module, HookPoint) and tag in module.tags:
                names.append(name)
        values = [self.data.activations[name].squeeze() for name in names]
        return values

    def get_activation_value_by_name(self, name: str) -> torch.Tensor:
        """
        returns the activation for a specific module name. The full name
        must be given, e.g. "layers.0.attention.mha.fused_softmax"
        """
        return self.data.activations[name]


class GPTInterpretabilityModule(BaseInterpretabilityModule):
    def __init__(
        self,
    ):
        model = GPT.from_pretrained("gpt2")
        super().__init__(model)
        self.initialise()

    def custom_dataloader(self, block_size: int = 100):
        data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'train.bin'))
        assert os.path.exists(data_path), "run `python3 shakespeare.py to prepare train.bin"
        data = np.memmap(data_path, dtype=np.uint16, mode='r')

        def get_batch():
            ix = torch.randint(len(data) - block_size, (1,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            return x, y
        
        def generator():
            while True:
                yield get_batch()
        
        return generator()

    def custom_forward(self, dataloader, model: torch.nn.Module) -> dict:
        data, target = next(dataloader)
        out, _ = model(data, target)
        d = {"data": data, "target": target, "out": out}
        return d

    def custom_loss(self, out: torch.tensor, target: torch.tensor) -> torch.tensor:
        loss = F.cross_entropy(out.view(-1, out.size(-1)), target.view(-1), ignore_index=-1)
        return loss

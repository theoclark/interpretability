Neel Nanda's Transformer Lens Library (https://github.com/neelnanda-io/TransformerLens) is great. It has infinitely more functionality, is maintained properly and has much more efficient code. I use it and love it and you should to.

So, why did I create this? TransformerLens works great if you're happy to use the models which are supported in the library. If you have another model, perhaps one you've trained yourself, or isn't a transformer-based LLM, or that just happens not to currently be supported on TransformerLens, then it's not that easy to work with, or might not be appropriate at all.

I put this together as a minimally invasive library that you can wrap around an existing model but still make interpretability relatively easy and straightforward. It aims to be flexible (can work with any PyTorch model) and easy to use. The price for flexibility is that the setup is a bit more involved than simply working inside a notebook.

It doesn't currently have much in the way of functionality but it's a platform that can be built on.

# Usage

1. clone the repository
2. install packages: `pip install -r requirements.txt`

### MinGPT Example

This repo contains support for Andrej Karpathy's MinGPT (https://github.com/karpathy/minGPT) for use as an example but the idea is that you can hook this up to any model.

1. Create the Shakespeare dataset: `python3 interpretability/gpt/shakespeare.py`
2. Open up the example notebook: `jupyter-lab introduction.ipynb`
3. Follow the examples and start hacking away

### Adapt to your own model

1. Create a new folder inside "/interpretability" and move across your existing model definition and relevant files
2. Inside `interpretability/module.py` create a child class of BaseModule that defines dataloader, forward and loss functions for your specific model. Use the pattern in `GPTModule` as guidance.
3. Inside your model definition create HookPoints in locations where you want to get activations. E.g:
    - At the top: `from .interpretability.utils import HookPoint`
    - Inside the relevant init method: `self.hook = HookPoint(tags=["attention"])`
    - Inside the relevant forward method: `x = self.hook(x)`
4. Inside a notebook, create an instance of your Module and off you go.


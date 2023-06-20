import torch
import numpy as np
import matplotlib.pyplot as plt
import circuitsvis as cv

device = "cuda" if torch.cuda.is_available() else "cpu"

class HookPoint(torch.nn.Module):
    """
    Identity function to be added inside model definition and targeted from the
    interpretability module via .get_activation_values_by_tag("tag_name")
    """
    def __init__(self, tags: list = None):
        super().__init__()
        self.tags = tags

    def forward(self, x):
        return x


def show_attention_head(attention_values, layer: int, head: int, **kwargs):
    """
    Takes a list of tensors of all the attention values from the model.
    Shows a visualisation of the attention pattern for a particular
    layer index and head index
    """
    attn = attention_values[layer][head, :, :]
    tokens = [str(i) for i in range(attn.shape[-1])]
    return cv.attention_pattern(tokens, attn, **kwargs)


def show_attention_heads(attention_values, layer: int, **kwargs):
    """
    Takes a list of tensors of all the attention values from the model.
    Shows a visualisation of the attention pattern for each head of a
    particular layer index.
    """
    tokens = [str(i) for i in range(attention_values[layer].shape[-1])]
    return cv.attention_heads(attention_values[layer], tokens, **kwargs)


def histogram(data, bins=100):
    """Plots a histogram"""
    counts, _bins = np.histogram(data, bins=bins)
    plt.stairs(counts, _bins)
    plt.show()

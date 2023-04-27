import jax

import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt


##
def torch_shape_get(input):
    def h_shape_get(x):
        return x.dtype, x.shape

    return jax.tree_map(h_shape_get, input)


##
class TorchModelMode:
    """
    A context manager to temporarily set a PyTorch model's mode to either training or evaluation, and restore its
    original mode when exiting the context.

    Args:
        model (torch.nn.Module): The PyTorch model whose mode needs to be temporarily set.
        mode (str): The mode to set the model to within the context. Must be either 'train' or 'eval'.

    Example:
        model = nn.Linear(10, 5)  # A simple PyTorch model for demonstration

        with TorchModelMode(model, 'eval'):
            # The model is in evaluation mode within this block
            pass

        # The model is back to its original mode after the block
    """

    def __init__(self, model: nn.Module, mode: str):
        self.model = model
        self.mode = mode
        self.original_mode = None

    def __enter__(self):
        self.original_mode = self.model.training  # Save the original mode
        if self.mode.lower() == "train":
            self.model.train()
        elif self.mode.lower() == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Invalid mode '{self.mode}', must be 'train' or 'eval'")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_mode:  # Restore the original mode
            self.model.train()
        else:
            self.model.eval()


##
torch_to_PIL = torchvision.transforms.ToPILImage()

def img_tensor_show(img_tensor):
    plt.imshow(torch_to_PIL(img_tensor))
##

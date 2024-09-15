"""Main script to demonstrate low-rank approximation of a neural network."""
import torch.nn as nn
from eigen_guess.low_rank import ModuleLowRank


def create_sample_model():
    """Create a sample neural network with linear layers."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model

def print_model_summary(model):
    """Print the summary of the model."""
    print("Model summary:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")

def main():
    """Main function."""
    model = create_sample_model()

    print("Original model:")
    print_model_summary(model)

    # Apply low-rank approximation
    low_rank = ModuleLowRank()
    model = low_rank(model)

    print("\nModel after applying low-rank approximation:")
    print_model_summary(model)


import torch
import deeptrack as dt
import numpy as np

# Define the scatterer with Torch functionality
scatterer = dt.PointParticle(position=(32, 32))
torch_scatterer = scatterer.torch()

# Create a PyTorch tensor to simulate the input
input_tensor = torch.tensor(np.zeros((1, 1, 64, 64)), dtype=torch.float32, requires_grad=True)

# Define the microscope optics (e.g., Fluorescence)
optics = dt.optics.Fluorescence(
    NA=0.7,
    wavelength=680e-9,
    resolution=1e-6,
    magnification=10,
    output_region=(0, 0, 64, 64),
)

# Simulate imaging using the PyTorch-compatible feature
imaged_scatterer = optics(torch_scatterer)

# Perform a backward pass to compute gradients
output_tensor = imaged_scatterer(input_tensor)
output_tensor.backward()

# The gradients are stored in input_tensor.grad
print(input_tensor.grad)
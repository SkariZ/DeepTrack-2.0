
import torch
import deeptrack as dt

optics = dt.optics_torch.Brightfield(
    NA=0.7,
    wavelength=680e-9,
    resolution=1e-6,
    magnification=10,
    output_region=(0, 0, 64, 64),
)

# Random 64*64 image
image = torch.rand(1, 64, 64)
image = dt.Image(image)

imaged_scatterer = optics(image)
imaged_scatterer.update()()


microscope = dt.Microscope(sample=image, objective=optics)
image = microscope.get(None)
print(image.shape)

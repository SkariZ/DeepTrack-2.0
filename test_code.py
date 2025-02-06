
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
image = torch.rand(1, 64, 64, 64)
image = dt.Image(image)

imaged_scatterer = optics(image)
imaged_scatterer.update()()


microscope = dt.Microscope(sample=image, objective=optics)
image = microscope.get(None)
print(image.shape)


def setup_optics(nsize, wavelength=532e-9, resolution=100e-9, magnification=1, return_field=True):
    """
    Set up the optical system, prepare simulation parameters, and compute the optical image.

    Args:
        nsize (int): Size of the volume grid.
        wavelength (float): Wavelength of light in meters (default 532 nm).
        resolution (float): Optical resolution in meters (default 100 nm).
        magnification (float): Magnification factor (default 1).
        return_field (bool): Whether to return the optical field (default True).

    Returns:
        dict: A dictionary containing optics object, limits, fields, properties, and computed image.
    """

    # Define the optics
    optics = dt.optics_torch.Brightfield(
        wavelength=wavelength,
        resolution=resolution,
        magnification=magnification,
        output_region=(0, 0, nsize, nsize),
        return_field=return_field
    )

    # Define simulation limits
    limits = torch.tensor([[0, nsize], [0, nsize], [-nsize//2, nsize//2]])

    # Define fields
    padded_nsize = ((nsize + 31) // 32) * 32
    fields = torch.ones((padded_nsize, padded_nsize), dtype=torch.complex64)

    # Extract relevant properties from the optics
    properties = optics.properties()
    filtered_properties = {
        k: v for k, v in properties.items()
        if k in {'padding', 'output_region', 'NA', 'wavelength', 
                 'refractive_index_medium', 'return_field'}
    }

    return {
        "optics": optics,
        "limits": limits,
        "fields": fields,
        "filtered_properties": filtered_properties,
        }

optics_setup = setup_optics(64)
object = dt.Image(object)
image = optics_setup['optics'].get(object, optics_setup['limits'], optics_setup['fields'], **optics_setup['filtered_properties'])
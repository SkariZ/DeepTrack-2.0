"""Features for optical imaging of samples.

This module provides classes and functionalities for simulating optical
imaging systems, enabling the generation of realistic camera images of
biological and physical samples. The primary goal is to offer tools for
modeling and computing optical phenomena such as brightfield, fluorescence,
holography, and other imaging modalities.

Key Features
------------
- **Microscope Simulation**

  The `Microscope` class acts as a high-level interface for imaging samples
  using defined optical systems. It coordinates the interaction between the
  sample and the optical system, ensuring seamless simulation of imaging
  processes.

- **Optical Systems**

  The `Optics` class and its derived classes represent various optical
  devices, defining core imaging properties such as resolution, magnification,
  numerical aperture (NA), and wavelength. Subclasses like `Brightfield`,
  `Fluorescence`, `Holography`, `Darkfield`, and `ISCAT` offer specialized
  configurations tailored to different imaging techniques.

- **Sample Illumination and Volume Simulation**

  Features like `IlluminationGradient` enable realistic simulation of
  non-uniform sample illumination, critical for generating realistic images.
  The `_create_volume` function facilitates combining multiple scatterers
  into a single unified volume, supporting 3D imaging.

- **Integration with DeepTrack**

  Full compatibility with DeepTrack's feature pipeline allows for dynamic
  and complex simulations, incorporating physics-based models and real-time
  adjustments to sample and imaging properties.

Module Structure
----------------
Classes:

- `Microscope`: Represents a simulated optical microscope that integrates the 
sample and optical systems. It provides an interface to simulate imaging by 
combining the sample properties with the configured optical system.

- `Optics`: An abstract base class representing a generic optical device. 
Subclasses implement specific optical systems by defining imaging properties 
and behaviors.

- `Brightfield`:  Simulates brightfield microscopy, commonly used for observing
unstained or stained samples under transmitted light. This class serves as the 
base for additional imaging techniques.

- `Holography`: Simulates holographic imaging, capturing phase information from
the sample. Suitable for reconstructing 3D images and measuring refractive 
index variations.  

- `Darkfield`: Simulates darkfield microscopy, which enhances contrast by 
imaging scattered light against a dark background. Often used to highlight fine
structures in samples.  

- `ISCAT`: Simulates interferometric scattering microscopy (ISCAT), an advanced 
technique for detecting small particles or molecules based on scattering and 
interference.  

- `Fluorescence`: Simulates fluorescence microscopy, modeling emission 
processes for fluorescent samples. Includes essential optical system 
configurations and fluorophore behavior.

- `IlluminationGradient`: Adds a gradient to the illumination of the sample, 
enabling simulations of non-uniform lighting conditions often seen in 
real-world experiments.

Utility Functions:

- `_get_position(image, mode, return_z)`

    def _get_position(
        image: np.ndarray, mode: str = "corner", return_z: bool = False
    ) -> Tuple[int, int, Optional[int]]

    Extracts the position of the upper-left corner of a scatterer in the image.

- `_create_volume(list_of_scatterers:, pad, output_region, refractive_index_medium, **kwargs)`

    def _create_volume(
        list_of_scatterers: List[np.ndarray],
        pad: int,
        output_region: Tuple[int, int, int, int],
        refractive_index_medium: float,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray

    Combines multiple scatterer objects into a single 3D volume for imaging.

- `_pad_volume(volume, limits, padding, output_region, **kwargs)`

    def _pad_volume(
        volume: np.ndarray,
        limits: np.ndarray,
        padding: Tuple[int, int, int, int],
        output_region: Tuple[int, int, int, int],
        **kwargs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]

    Pads a volume with zeros to avoid edge effects during imaging.

Examples
--------
Simulating an image with the `Brightfield` class:

>>> import deeptrack as dt

>>> scatterer = dt.PointParticle()
>>> optics = dt.Brightfield()
>>> image = optics(scatterer)
>>> print(image().shape)
(128, 128, 1)
>>> image.plot(cmap="gray")

Simulating an image with the `Fluorescence` class:

>>> import deeptrack as dt

>>> scatterer = dt.PointParticle()
>>> optics = dt.Fluorescence()
>>> image = optics(scatterer)
>>> print(image().shape)
(128, 128, 1)
>>> image.plot(cmap="gray")

"""

from pint import Quantity
from typing import Any, Dict, List, Tuple, Union
from deeptrack.backend.units import (
    ConversionTable,
    create_context,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.math import AveragePooling
from deeptrack.features import propagate_data_to_dependencies
import numpy as np
from .features import DummyFeature, Feature, StructuralFeature
from .image import Image, pad_image_to_fft, maybe_cupy
from .types import ArrayLike, PropertyLike
from .backend._config import cupy
from scipy.ndimage import convolve
import warnings

from . import units as u
from .backend import config
from deeptrack import image


import torch
from typing import Any, Dict, Union

class Microscope(StructuralFeature):
    """Simulates imaging of a sample using an optical system using PyTorch.
    """
    __distributed__ = False

    def __init__(
        self: 'Microscope',
        sample: Feature,
        objective: Feature,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self._sample = self.add_feature(sample)
        self._objective = self.add_feature(objective)
        self._sample.store_properties()

    def get(
        self: 'Microscope',
        image: Union[Image, None],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Generate an image of the sample using the defined optical system.
        """
        additional_sample_kwargs = self._objective.properties()
        _upscale_given_by_optics = additional_sample_kwargs["upscale"]
        if torch.tensor(_upscale_given_by_optics).numel() == 1:
            _upscale_given_by_optics = (_upscale_given_by_optics,) * 3

        upscale = torch.round(get_active_scale())

        output_region = additional_sample_kwargs.pop("output_region")
        additional_sample_kwargs["output_region"] = [
            int(o * upsc)
            for o, upsc in zip(
                output_region, (upscale[0], upscale[1], upscale[0], upscale[1])
            )
        ]

        padding = additional_sample_kwargs.pop("padding")
        additional_sample_kwargs["padding"] = [
            int(p * upsc)
            for p, upsc in zip(
                padding, (upscale[0], upscale[1], upscale[0], upscale[1])
            )
        ]

        self._objective.output_region.set_value(
            additional_sample_kwargs["output_region"]
        )
        self._objective.padding.set_value(additional_sample_kwargs["padding"])

        propagate_data_to_dependencies(
            self._sample, **{"return_fft": True, **additional_sample_kwargs}
        )

        list_of_scatterers = self._sample()
        if not isinstance(list_of_scatterers, list):
            list_of_scatterers = [list_of_scatterers]

        volume_samples = [
            scatterer for scatterer in list_of_scatterers
            if not scatterer.get_property("is_field", default=False)
        ]
        
        field_samples = [
            scatterer for scatterer in list_of_scatterers
            if scatterer.get_property("is_field", default=False)
        ]

        sample_volume, limits = _create_volume(
            volume_samples,
            **additional_sample_kwargs,
        )
        sample_volume = Image(sample_volume)

        for scatterer in volume_samples + field_samples:
            sample_volume.merge_properties_from(scatterer)

        propagate_data_to_dependencies(
            self._objective,
            limits=limits,
            fields=field_samples,
        )

        imaged_sample = self._objective.resolve(sample_volume)

        if _upscale_given_by_optics != (1, 1, 1):
            imaged_sample = torch.nn.functional.avg_pool2d(
                imaged_sample.unsqueeze(0).unsqueeze(0),
                kernel_size=(*_upscale_given_by_optics[:2], 1)
            ).squeeze(0).squeeze(0)

        if not image:
            return imaged_sample if self._wrap_array_with_image else imaged_sample._value

        if not isinstance(image, list):
            image = [image]
        for i in range(len(image)):
            image[i].merge_properties_from(imaged_sample)
        return image


import torch
import torch.nn as nn
import torch.fft as fft
import warnings
import numpy as np

class Optics(nn.Module):
    def __init__(self, NA=0.7, wavelength=0.66e-6, magnification=10, resolution=1e-6,
                 refractive_index_medium=1.33, padding=(10, 10, 10, 10),
                 output_region=(0, 0, 128, 128), upscale=1):
        super(Optics, self).__init__()
        
        self.NA = NA
        self.wavelength = wavelength
        self.magnification = magnification
        self.resolution = resolution
        self.refractive_index_medium = refractive_index_medium
        self.padding = padding
        self.output_region = output_region
        self.upscale = upscale

    def get_voxel_size(self):
        return torch.tensor(self.resolution / self.magnification, dtype=torch.float32)
    
    def get_pixel_size(self):
        return self.get_voxel_size()

    def _pupil(self, shape, NA, wavelength, refractive_index_medium, include_aberration=True, defocus=0):
        shape = torch.tensor(shape, dtype=torch.float32)
        voxel_size = self.get_voxel_size()

        R = NA / wavelength * voxel_size[:2]
        x_radius, y_radius = R * shape[:2]

        x = torch.linspace(-shape[0] / 2, shape[0] / 2 - 1, int(shape[0])) / x_radius
        y = torch.linspace(-shape[1] / 2, shape[1] / 2 - 1, int(shape[1])) / y_radius

        W, H = torch.meshgrid(y, x, indexing='ij')
        RHO = (W ** 2 + H ** 2).to(torch.complex64)
        pupil_function = (RHO < 1).to(torch.complex64)

        z_shift = (2 * np.pi * refractive_index_medium / wavelength * voxel_size[2] * 
                   torch.sqrt(1 - (NA / refractive_index_medium) ** 2 * RHO))
        z_shift[z_shift.isnan()] = 0
        
        defocus = torch.tensor(defocus).reshape(-1, 1, 1)
        pupil_functions = pupil_function * torch.exp(1j * z_shift * defocus)
        return pupil_functions

    def _pad_volume(self, volume, limits, padding, output_region):
        limits = torch.tensor(limits)
        output_region = torch.tensor(output_region)
        new_limits = limits.clone()
        
        for i in range(2):
            new_limits[i, 0] = min(new_limits[i, 0], output_region[i] - padding[i])
            new_limits[i, 1] = max(new_limits[i, 1], output_region[i + 2] + padding[i + 2])
        
        new_shape = (new_limits[:, 1] - new_limits[:, 0]).tolist()
        new_volume = torch.zeros(new_shape, dtype=torch.complex64)
        
        old_region = (limits - new_limits).int()
        new_volume[
            old_region[0, 0]: old_region[0, 0] + (limits[0, 1] - limits[0, 0]),
            old_region[1, 0]: old_region[1, 0] + (limits[1, 1] - limits[1, 0]),
            old_region[2, 0]: old_region[2, 0] + (limits[2, 1] - limits[2, 0])
        ] = volume
        return new_volume, new_limits

    def forward(self, sample):
        return sample


class Fluorescence(Optics):
    """PyTorch-optimized Optical device for fluorescent imaging."""

    __gpu_compatible__ = True

    def get(
        self, 
        illuminated_volume: torch.Tensor, 
        limits: np.ndarray, 
        **kwargs
    ) -> Image:
        """Simulates the imaging process using a fluorescence microscope."""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to PyTorch tensor and move to GPU
        illuminated_volume = torch.tensor(illuminated_volume, dtype=torch.complex64).to(device)

        # Pad volume
        padded_volume, limits = self._pad_volume(illuminated_volume, limits=limits, **kwargs)

        # Extract output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))

        # Calculate cropping
        output_region[0] = (
            None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        )

        padded_volume = padded_volume[
            output_region[0]:output_region[2],
            output_region[1]:output_region[3],
            :,
        ]
        z_limits = limits[2, :]

        output_image = torch.zeros((*padded_volume.shape[0:2], 1), device=device)

        # Optimize by skipping empty planes
        z_iterator = torch.linspace(
            z_limits[0], z_limits[1], num=padded_volume.shape[2], device=device
        )
        zero_plane = torch.all(padded_volume == 0, dim=(0, 1))
        z_values = z_iterator[~zero_plane]

        # Pad image for FFT optimization
        volume = pad_image_to_fft(padded_volume, axes=(0, 1))
        volume = torch.tensor(volume, dtype=torch.complex64, device=device)

        pupils = self._pupil(volume.shape[:2], defocus=z_values, **kwargs)

        z_index = 0
        for i, z in enumerate(z_iterator):

            if zero_plane[i]:
                continue

            pupil = pupils[z_index].to(device)
            z_index += 1

            psf = torch.fft.ifft2(torch.fft.fftshift(pupil)).abs()**2
            otf = torch.fft.fft2(psf)
            fourier_field = torch.fft.fft2(volume[:, :, i])
            convolved_fourier_field = fourier_field * otf
            field = torch.fft.ifft2(convolved_fourier_field).real

            output_image[:, :, 0] += field[:padded_volume.shape[0], :padded_volume.shape[1]]

        output_image = output_image[pad[0]:-pad[2], pad[1]:-pad[3]]

        #output_image = Image(output_image.cpu().numpy(), copy=False)
        output_image = Image(output_image, copy=False)

        output_image.properties = illuminated_volume.properties + pupils.properties

        return output_image


import torch
import numpy as np
from deeptrack.optics import Optics
from deeptrack.image import Image
from deeptrack.utils import maybe_cupy, pad_image_to_fft

class Brightfield(Optics):
    """GPU-accelerated Brightfield microscopy simulation using PyTorch."""

    __gpu_compatible__ = True

    def get(
        self, 
        illuminated_volume: torch.Tensor, 
        limits: np.ndarray, 
        fields: torch.Tensor, 
        **kwargs
    ) -> Image:
        """Simulates imaging with brightfield microscopy using PyTorch."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to PyTorch tensor and move to GPU
        illuminated_volume = torch.tensor(illuminated_volume, dtype=torch.complex64).to(device)

        # Pad volume
        padded_volume, limits = self._pad_volume(illuminated_volume, limits=limits, **kwargs)

        # Extract output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))

        output_region[0] = (
            None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        )

        padded_volume = padded_volume[
            output_region[0]:output_region[2],
            output_region[1]:output_region[3],
            :,
        ]
        z_limits = limits[2, :]

        output_image = torch.zeros((*padded_volume.shape[0:2], 1), device=device)

        z_iterator = torch.linspace(z_limits[0], z_limits[1], num=padded_volume.shape[2], device=device)
        zero_plane = torch.all(padded_volume == 0, dim=(0, 1))
        
        volume = pad_image_to_fft(padded_volume.cpu().numpy(), axes=(0, 1))
        volume = torch.tensor(volume, dtype=torch.complex64, device=device)

        voxel_size = self.get_active_voxel_size()
        K = 2 * torch.pi / kwargs["wavelength"] * kwargs["refractive_index_medium"]

        # Pupil functions
        pupils = [
            torch.tensor(self._pupil(volume.shape[:2], defocus=[1], include_aberration=False, **kwargs)[0], dtype=torch.complex64, device=device),
            torch.tensor(self._pupil(volume.shape[:2], defocus=[-z_limits[1]], include_aberration=True, **kwargs)[0], dtype=torch.complex64, device=device),
            torch.tensor(self._pupil(volume.shape[:2], defocus=[0], include_aberration=True, **kwargs)[0], dtype=torch.complex64, device=device)
        ]

        pupil_step = torch.fft.fftshift(pupils[0])

        light_in = torch.ones(volume.shape[:2], dtype=torch.complex64, device=device)
        light_in = torch.fft.fft2(light_in)

        z_index = 0
        for i, z in enumerate(z_iterator):
            light_in = light_in * pupil_step

            if zero_plane[i]:
                continue

            ri_slice = volume[:, :, i]
            light = torch.fft.ifft2(light_in)
            light_out = light * torch.exp(1j * ri_slice * voxel_size[-1] * K)
            light_in = torch.fft.fft2(light_out)

        shifted_pupil = torch.fft.fftshift(pupils[1])
        light_in_focus = light_in * shifted_pupil

        if fields.shape[0] > 0:
            field = torch.sum(fields, dim=0).to(device)
            light_in_focus += field[..., 0]

        shifted_pupil = torch.fft.fftshift(pupils[-1])
        light_in_focus = light_in_focus * shifted_pupil

        mask = torch.abs(shifted_pupil) > 0
        light_in_focus = light_in_focus * mask

        output_image = torch.fft.ifft2(light_in_focus).real
        output_image = output_image[:padded_volume.shape[0], :padded_volume.shape[1]]
        output_image = output_image.unsqueeze(-1)

        #output_image = Image(output_image[pad[0]:-pad[2], pad[1]:-pad[3]].cpu().numpy())
        output_image = Image(output_image[pad[0]:-pad[2], pad[1]:-pad[3]])

        if not kwargs.get("return_field", False):
            output_image = torch.square(torch.abs(output_image))

        output_image.properties = illuminated_volume.properties

        return output_image



class Holography(Brightfield):
    """An alias for the Brightfield class, representing holographic 
    imaging setups.

    Holography shares the same implementation as Brightfield, as both use 
    coherent illumination and similar propagation techniques.

    """
    pass


class ISCAT(Brightfield):
    """Images coherently illuminated samples using Interferometric Scattering 
    (ISCAT) microscopy.

    This class models ISCAT by creating a discretized volume where each pixel
    represents the effective refractive index of the sample. Light is 
    propagated through the sample iteratively, first in the Fourier space 
    and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π radians.
    amp_factor: float, optional
        Amplitude factor of the illuminating field relative to the reference 
        field. Default is 1.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.
    amp_factor: float
        Amplitude factor of the illuminating field.

    Examples
    --------
    Creating an ISCAT instance:
    
    >>> import deeptrack as dt

    >>> iscat = dt.ISCAT(NA=1.4, wavelength=0.532e-6, magnification=60)
    >>> print(iscat.illumination_angle())
    3.141592653589793
    
    """

    def __init__(
        self:  'ISCAT',
        illumination_angle: float = torch.pi,
        amp_factor: float = 1, 
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the ISCAT class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        amp_factor: float
            Amplitude factor of the illuminating field relative to the reference 
            field.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            amp_factor=amp_factor,
            input_polarization="circular",
            output_polarization="circular",
            phase_shift_correction=True,
            **kwargs
            )
        
class Darkfield(Brightfield):
    """Images coherently illuminated samples using Darkfield microscopy.

    This class models Darkfield microscopy by creating a discretized volume 
    where each pixel represents the effective refractive index of the sample. 
    Light is propagated through the sample iteratively, first in the Fourier 
    space and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π/2 radians.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.

    Methods
    -------
    get(illuminated_volume, limits, fields, **kwargs)
        Retrieves the darkfield image of the illuminated volume.

    Examples
    --------
    Creating a Darkfield instance:

    >>> import deeptrack as dt

    >>> darkfield = dt.Darkfield(NA=0.9, wavelength=0.532e-6)
    >>> print(darkfield.illumination_angle())
    1.5707963267948966

    """

    def __init__(
        self: 'Darkfield', 
        illumination_angle: float = np.pi/2, 
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the Darkfield class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            **kwargs)

    #Retrieve get as super
    def get(
        self: 'Darkfield',
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Retrieve the darkfield image of the illuminated volume.

        Parameters
        ----------
        illuminated_volume: array_like
            The volume of the sample being illuminated.
        limits: array_like
            The spatial limits of the volume.
        fields: array_like
            The fields interacting with the sample.
        **kwargs: Dict[str, Any]
            Additional parameters passed to the super class's get method.

        Returns
        -------
        numpy.ndarray
            The darkfield image obtained by calculating the squared absolute
            difference from 1.
        
        """

        field = super().get(illuminated_volume, limits, fields, return_field=True, **kwargs)
        return torch.square(torch.abs(field-1))

import torch

def _get_position(
    image: Image,  # Expecting an Image-like object with .to_tensor() and .get_property()
    mode: str = "corner",
    return_z: bool = False,
) -> torch.Tensor:
    """Extracts the position of the upper-left corner of a scatterer.

    Parameters
    ----------
    image: torch.Tensor or Image-like object
        Input image or volume containing the scatterer.
    mode: str, optional
        Mode for position extraction. Default is "corner".
    return_z: bool, optional
        Whether to include the z-coordinate in the output. Default is False.

    Returns
    -------
    torch.Tensor
        Tensor containing the position of the scatterer.
    
    """

    num_outputs = 2 + return_z

    if mode == "corner" and image.numel() > 0:
        image_tensor = image.to_tensor()  # Assuming `image` has a `to_tensor()` method
        abs_image = torch.abs(image_tensor)
        
        # Compute center of mass manually since PyTorch lacks an equivalent function
        indices = torch.nonzero(abs_image)
        if indices.numel() > 0:
            shift = indices.float().mean(dim=0)
        else:
            shift = torch.tensor(image_tensor.shape, dtype=torch.float32) / 2
    else:
        shift = torch.zeros(num_outputs)

    position = image.get_property("position", default=None)

    if position is None:
        return position

    position = torch.tensor(position, dtype=torch.float32)
    scale = torch.tensor(get_active_scale(), dtype=torch.float32)  # Assuming `get_active_scale()` returns a list/array

    if position.shape[0] == 3:
        position = position * scale + 0.5 * (scale - 1)
        if return_z:
            return position * scale - shift
        else:
            return position[:2] - shift[:2]

    elif position.shape[0] == 2:
        if return_z:
            z_val = image.get_property("z", default=0)
            z_val = torch.tensor(z_val, dtype=torch.float32)
            outp = torch.tensor([position[0], position[1], z_val]) * scale - shift + 0.5 * (scale - 1)
            return outp
        else:
            return position * scale[:2] - shift[:2] + 0.5 * (scale[:2] - 1)

    return position


import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

def _create_volume(
    list_of_scatterers: List,
    pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
    output_region: Tuple[int, int, int, int] = (None, None, None, None),
    refractive_index_medium: float = 1.33,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if not isinstance(list_of_scatterers, list):
        list_of_scatterers = [list_of_scatterers]
    
    volume = torch.zeros((1, 1, 1), dtype=torch.cfloat)
    limits = None
    OR = torch.tensor([
        float('inf') if output_region[0] is None else output_region[0] - pad[0],
        float('-inf') if output_region[1] is None else output_region[1] - pad[1],
        float('inf') if output_region[2] is None else output_region[2] + pad[2],
        float('-inf') if output_region[3] is None else output_region[3] + pad[3],
    ])
    
    scale = torch.tensor(get_active_scale())
    fudge_factor = scale[0] * scale[1] / scale[2]
    
    for scatterer in list_of_scatterers:
        position = _get_position(scatterer, mode="corner", return_z=True)
        
        if scatterer.get_property("intensity", None) is not None:
            scatterer_value = scatterer.get_property("intensity") * fudge_factor
        elif scatterer.get_property("refractive_index", None) is not None:
            scatterer_value = scatterer.get_property("refractive_index") - refractive_index_medium
        else:
            scatterer_value = scatterer.get_property("value")
        
        scatterer = scatterer * scatterer_value
        
        if limits is None:
            limits = torch.zeros((3, 2), dtype=torch.int32)
            limits[:, 0] = torch.floor(position).int()
            limits[:, 1] = torch.floor(position).int() + 1
        
        if (
            position[0] + scatterer.shape[0] < OR[0]
            or position[0] > OR[2]
            or position[1] + scatterer.shape[1] < OR[1]
            or position[1] > OR[3]
        ):
            continue
        
        scatterer = F.pad(scatterer, (2, 2, 2, 2, 2, 2), "constant", 0)
        position = _get_position(scatterer, mode="corner", return_z=True)
        shape = torch.tensor(scatterer.shape)
        
        if position is None:
            continue
        
        splined_scatterer = torch.zeros_like(scatterer)
        x_off, y_off = position[0] % 1, position[1] % 1
        kernel = torch.tensor([
            [0, 0, 0],
            [0, (1 - x_off) * (1 - y_off), (1 - x_off) * y_off],
            [0, x_off * (1 - y_off), x_off * y_off],
        ]).unsqueeze(0).unsqueeze(0)
        
        for z in range(scatterer.shape[2]):
            if splined_scatterer.dtype == torch.cfloat:
                real_part = F.conv2d(scatterer[:, :, z].real.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                imag_part = F.conv2d(scatterer[:, :, z].imag.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                splined_scatterer[:, :, z] = real_part.squeeze() + 1j * imag_part.squeeze()
            else:
                splined_scatterer[:, :, z] = F.conv2d(scatterer[:, :, z].unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()
        
        scatterer = splined_scatterer
        position = torch.floor(position)
        new_limits = torch.zeros_like(limits)
        for i in range(3):
            new_limits[i, :] = torch.tensor([
                min(limits[i, 0], position[i]),
                max(limits[i, 1], position[i] + shape[i])
            ], dtype=torch.int32)
        
        if not torch.equal(new_limits, limits):
            new_volume_shape = (new_limits[:, 1] - new_limits[:, 0]).int()
            new_volume = torch.zeros(new_volume_shape.tolist(), dtype=torch.cfloat)
            old_region = (limits - new_limits).int()
            
            new_volume[
                old_region[0, 0] : old_region[0, 0] + (limits[0, 1] - limits[0, 0]),
                old_region[1, 0] : old_region[1, 0] + (limits[1, 1] - limits[1, 0]),
                old_region[2, 0] : old_region[2, 0] + (limits[2, 1] - limits[2, 0]),
            ] = volume
            
            volume = new_volume
            limits = new_limits
        
        within_volume_position = position - limits[:, 0]
        
        volume[
            int(within_volume_position[0]) : int(within_volume_position[0] + shape[0]),
            int(within_volume_position[1]) : int(within_volume_position[1] + shape[1]),
            int(within_volume_position[2]) : int(within_volume_position[2] + shape[2]),
        ] += scatterer
    
    return volume, limits
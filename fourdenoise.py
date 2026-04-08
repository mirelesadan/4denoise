"""
The 4denoise data structures:
    - HyperData
    - ReciprocalSpace
    - RealSpace
    - DenoisingMethods
    - Denoiser

Author: 
    Adan J Mireles
    Smalley Curl Insitute, Applied Physics
    Department of Materials Science and Nanoengineering
    Rice University; Houston, TX 

Date:
    April 2024
"""

import numpy as np
import h5py
from numbers import Integral


from scipy import ndimage
from scipy.stats import mode
from scipy.signal import find_peaks as hist_peaks
import scipy.stats
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage import rotate
from scipy.ndimage import grey_erosion, grey_dilation
from scipy import io
from scipy.linalg import polar
from scipy.fft import fft2, fftshift
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.ndimage import affine_transform
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import time

from skimage.measure import profile_line
from skimage.feature import peak_local_max
from skimage import transform
from skimage import feature
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
import inspect

import numba #new
from numba import jit, prange #new

# import tensorly as tl

# from tensorly.decomposition import constrained_parafac
# from tensorly.decomposition import parafac2 as par2
# from tensorly.decomposition import tensor_ring as t_ring
# from tensorly.decomposition import tensor_train_matrix as tt_mat
# from tensorly.decomposition import tensor_train as tt
# from tensorly.decomposition import non_negative_tucker_hals as nnth
# from tensorly.decomposition import non_negative_tucker as nnt
# from tensorly.decomposition import partial_tucker as partial_tuck
# from tensorly.decomposition import tucker as tuck
# from tensorly.decomposition import randomised_parafac as rand_parafac
# from tensorly.decomposition import non_negative_parafac_hals as nn_parafac_hals
# from tensorly.decomposition import non_negative_parafac as nn_parafac
# from tensorly.decomposition import parafac as par

# from tensorly.decomposition import parafac_power_iteration as parafac_power_iter
# from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac_power_iter

from typing import Union, Sequence, Tuple
    
#%%

# =============================================================================
# Useful Functions
# =============================================================================

#TODO: Allow for reading data that has more than (128,128) in k-space
#TODO: read EMD file data

def read_4D(fname, dp_dims=(128, 130), trim_dims=(128,128), trim_meta=True, clip=True):
    """
    Read the 4D dataset as a numpy array from .raw , . mat, .npy file.
    
    Function written by Chuqiao Shi (2022)
    See on GitHub: Chuqiao2333/Hierarchical_Clustering
    
    Modified by Adan J Mireles (April, 2024)
    - Addition of 'dp_dims', 'trim_dims', and 'trim_meta'
    - Modified 'print' statement
    
    Input:
        fname: the file path

    Return: 
        dp       : numpy array
        dp_shape : the shape of the data
        
    Function modified by Adan Mireles to make '.mat' file reading more general (July 2024)
    """

    # Read 4D data from .raw file
    fname_end = fname.split('.')[-1]
        
    if fname_end == 'raw':
        with open(fname, 'rb') as file:
            dp = np.fromfile(file, np.float32)

        columns = dp_dims[0]    
        rows = dp_dims[1]
            
        sqpix = dp.size/columns/rows
        
        # Assuming square scan, i.e. same number of x and y scan points
        pix = int(sqpix**(0.5))
        
        dp = np.reshape(dp, (pix, pix, rows, columns), order = 'C')
        
        # Trim off the last two meta data rows if desired. The metadata is for EMPAD debugging, and generally doesn't need to be kept.
        if trim_meta:
            dp = dp[:,:,:trim_dims[0],:trim_dims[1]]

    elif fname_end == 'mat':
        dp = _read_mat_file(fname)
        
    elif fname_end == 'npy':
        dp = np.load(fname)
        
    else:
        print('This function only supports reading .mat, .raw & .npy files.') 
        
    if clip:
        # Replace negative and near-zero pixel values with 1
        low_vals_mask = dp < 1
        dp[low_vals_mask] = 1
    
    # We proceed to determine whether there are any `nan` values in the dataset
    a, b, c, d = dp.shape
    nan_mask = np.isnan(dp).any(axis=(2, 3))  # shape (a, b)
    
    num_NaNs = nan_mask.sum()
    if num_NaNs > 0:
        print(f'Found {num_NaNs} corrupted diffraction patterns. Replacing with local average...')
        
        # Find all indices of bad diffraction patterns
        bad_indices = np.argwhere(nan_mask)
        
        # Iterate over each bad diffraction pattern
        for i, j in bad_indices:
            neighbors = []
            # loop over 8-connected neighborhood
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    # check bounds and that neighbor has no NaNs
                    if 0 <= ni < a and 0 <= nj < b and not nan_mask[ni, nj]:
                        neighbors.append(dp[ni, nj])
            
            # if valid neighbors found, replace with their mean
            if neighbors:
                stack = np.stack(neighbors, axis=0)
                dp[i, j] = stack.mean(axis=0)
            else:
                # no valid neighbor, you may choose to leave it or fill zeros
                dp[i, j] = 0.0 
                
        print('...Done')
        
    return dp

def _read_mat_file(filename):
    """Read and extract data from a .mat file using scipy.io.

    This function attempts to read a .mat file using the scipy.io.loadmat method. It searches for 
    the first key in the file that corresponds to a multi-dimensional numpy array and returns that 
    array. If the file cannot be read due to format issues, it attempts to load it using an 
    alternative method (_read_mat_file_h5py).

    Parameters
    ----------
    filename : str
        The path to the .mat file to be read.

    Returns
    -------
    content : np.ndarray or dict or None
        The extracted data as a numpy array if successful, a dictionary if no suitable array is 
        found, or None if reading the file fails.

    Examples
    --------
    >>> data = _read_mat_file('example.mat')
    >>> print(data.shape)
    (100, 100)

    Notes
    -----
    If the file is in HDF5 format, it will automatically try to load the file using the h5py method.
    """
    try:
        file = io.loadmat(filename)

        for key in file.keys():
            content = file[key]
            
            if isinstance(content, np.ndarray):
                if len(content.shape) > 1:
                    return content
        else:
            print("No suitable key found in the .mat file.")
            return file
        
    except NotImplementedError:
        print("This file may be in HDF5 format. Trying h5py to load.")
        return _read_mat_file_h5py(filename)
    
    except Exception as e:
        print(f"Failed to read the .mat file: {e}")
        return None


def _read_mat_file_h5py(filename):
    """Read and extract data from a .mat file using h5py.

    This function attempts to read a .mat file using the h5py library. It searches for the first 
    key in the file that corresponds to a multi-dimensional numpy array and returns that array. 
    If no suitable array is found, it returns the file object or None if reading the file fails.

    Parameters
    ----------
    filename : str
        The path to the .mat file to be read.

    Returns
    -------
    content : np.ndarray or h5py.File or None
        The extracted data as a numpy array if successful, the file object if no suitable array 
        is found, or None if reading the file fails.

    Notes
    -----
    This method is specifically used for .mat files that are stored in HDF5 format. It automatically 
    attempts to convert the HDF5 datasets into numpy arrays.
    """
    try:
        success_msg = "...Data loaded successfully."
        with h5py.File(filename, 'r') as file:
            
            for key in file.keys():
                content = file[key]
                
                if isinstance(content, np.ndarray):
                    if len(content.shape) > 1:
                        print(success_msg)
                        return content
                    
                elif len(content.shape) > 1:
                    print(success_msg)
                    return np.array(content)
            else:
                print("No suitable key found in the .mat file.")
                return file
            
    except Exception as e:
        print(f"Error reading with h5py: {e}")
        return None


def save_mat_data(data, fileName, varName):
    """
    Save a numpy array as .mat file.
    
    Inputs: 
    - data: numpy array
    - fileName: string with '.mat' termination
    - varName: string - name of the variable to appear on MATLAB
    """
    
    # Must verify that function has termination '.npy'
    io.savemat(fileName, {varName: data})


def circular_mask(center_y, center_x, radius):
    """
    Make a 2D boolean mask where 'True' corresponds to a circular region of 
    user-defined radius and position.
    
    Inputs:
        center_y: y-coordinate center of mask (int or float) 
        center_x: x-coordinate center of mask (int or float) 
        radius: radius of 'True'-valued values
    """ 
    
    y, x = np.ogrid[-center_y:radius*2-center_y, -center_x:radius*2-center_x]
    mask = x**2 + y**2 <= radius**2
    return mask


def make_mask(centers, r_mask, mask_dim=(128, 128), invert=False,):
    """
    Create a circular or annular mask around given center points.

    Generates a boolean mask of specified dimensions, marking pixels
    within a radius or between radii from the given center point(s).
    Optionally inverts the mask.

    Parameters
    ----------
    center : tuple or list of tuples
        A tuple (y, x) representing the center of the mask, or a list of such 
        tuples for multiple centers.
    r_mask : float or tuple of floats
        Radius of the mask. If a tuple (inner_radius, outer_radius) is provided, 
        an annular mask is created.
    mask_dim : tuple of ints, optional
        Dimensions of the mask (height, width). Default is (128, 128).
    invert : bool, optional
        If True, inverts the mask. Default is False.

    Returns
    -------
    mask : np.ndarray of bool
        The generated boolean mask.

    Notes
    -----
    The function assumes the origin (0, 0) is at the top-left corner.
    """

    mask = np.zeros(mask_dim, dtype=bool)
    centers = np.atleast_2d(np.array(centers))
    
    # Create vertical and horizontal vectors with indices
    y_grid, x_grid = np.ogrid[:mask_dim[0], :mask_dim[1]] 
    
    for center in centers:
        y, x = center
        
        # Get a 2D map of all combinations of distances
        dist_sq = (y_grid - y) ** 2 + (x_grid - x) ** 2
        
        if isinstance(r_mask, tuple):
            mask |= (r_mask[0]**2 <= dist_sq) & (dist_sq <= r_mask[1]**2)
        else:
            mask |= dist_sq <= r_mask**2

    if invert:
        mask = ~mask

    return mask

def anscombe_transform(array, inverse=False):
    """
    Function that will perform the forward Anscombe transform of input array
    to stabilize the variance of Poisson noise and make it a constant value.
    
    If 'inverse' is True, the inverse (unbiased) Anscombre transform is applied.
    
    We approximate the exact unbiased inverse transform using a closed-form
    expression. See Makitalo & Foi (2011); doi:10.1109/TIP.2011.2121085.
    """
    
    if not inverse:
        transformed_arr = 2*np.sqrt(array + 3/8)
        
    else:
        transformed_arr = (1/4)*array**2 + (1/4)*np.sqrt(3/2)*array**(-1) - \
                          (11/8)*array**(-2) + (5/8)*np.sqrt(3/2)*array**(-3) - 1/8
    
    return transformed_arr

def add_poisson_noise(array, counts):
    
    # Remove any negative numbers and normalize
    noisy_array = np.copy(array)
    noisy_array[noisy_array < 0] = 0
    noisy_array /= np.sum(noisy_array)
    
    # Apply Poisson noise
    noisy_array = np.random.default_rng().poisson(noisy_array * counts,) 
    
    return noisy_array

def get_surface_tilt_and_direction(surface, units='rad', show_results=True, figsize=(8, 5)):
    """
    Calculate the tilt magnitude and tilt axis (gradient direction) of a surface and optionally display the results.

    Parameters
    ----------
    surface : numpy.ndarray
        A 2D array of height values representing the surface.
    units : str, optional
        The unit of the tilt magnitude and direction. Can be 'rad' for radians or 'deg' for degrees.
        Default is 'rad'.
    show_results : bool, optional
        If True, plots the tilt magnitude and direction. Default is True.

    Returns
    -------
    tilt_magnitude : numpy.ndarray
        A 2D array where each value represents the angular magnitude of the gradient (tilt magnitude),
        in specified units (either radians or degrees).
    tilt_direction : numpy.ndarray
        A 2D array where each value represents the direction of the gradient, in specified units
        (either radians or degrees).


    Notes
    -----
    The tilt direction is calculated relative to the horizontal axis. Edge values for the tilt
    magnitude and direction are set based on the boundary conditions and may not accurately
    represent the actual tilt due to the lack of neighboring data.
    """
    
    # Compute gradients along x and y axes
    gy, gx = np.gradient(surface)
    
    # Calculate gradient magnitude and then calculate arctan of this magnitude
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    tilt_magnitude = np.arctan(gradient_magnitude)
    tilt_direction = np.arctan2(gy, gx) + np.pi
    
    # Convert tilt magnitude and direction to degrees if required
    if units == 'deg':
        tilt_magnitude = np.degrees(tilt_magnitude)
        tilt_direction = np.degrees(tilt_direction)
        unit_label = '(°)'
    else:
        unit_label = '(rad)'
    
    if show_results:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # First subplot
        im1 = axs[0].imshow(tilt_direction, cmap='hsv')
        axs[0].axis('off')
        axs[0].set_title('Azimuthal\nAngle', fontsize=8)
        cbar1 = fig.colorbar(im1, ax=axs[0],)
        cbar1.ax.set_title(r'$\phi$ '+unit_label, fontsize=8, pad=5)
        
        # Second subplot
        im2 = axs[1].imshow(tilt_magnitude, cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Elevation\nAngle', fontsize=8)
        cbar2 = fig.colorbar(im2, ax=axs[1],)
        cbar2.ax.set_title(r'$\theta$ '+unit_label, fontsize=8, pad=5)
        
        # Adjust layout to make the subplots fit well
        plt.tight_layout()
        plt.show()
    
    return tilt_direction, tilt_magnitude


def visualize_field_phase_amplitude(field, Amp='raw', scale=True, subplot=True):
    """
    Visualize the phase and amplitude of a field as an RGB image and optionally display a color wheel.

    Parameters
    ----------
    field : complex ndarray or tuple
        The input field. Can be a complex array or a tuple of (phase, magnitude).
    Amp : str, optional
        Control for amplitude visualization:
        'uniform' - use a uniform amplitude across the image,
        'log' - use logarithmic scaling of the amplitude,
        'raw' - use the raw amplitude values.
    scale : bool, optional
        If True, plots a color wheel with corresponding phase hues. Default is False.
    subplot : bool, optional
        If True, uses subplots to show the image and color wheel; otherwise, separate figures.

    Returns
    -------
    rgb_image : ndarray
        An array representing the RGB visualization of the input field.
    
    Notes
    -----
    This code is based on the MATLAB code written by [author] in [date].
    """
    
    if isinstance(field, tuple):
        phase, amplitude = field
    else:
        phase = np.angle(field)
        amplitude = np.abs(field)
    
    # Normalize amplitude
    amplitude = amplitude/np.max(amplitude)
    
    if Amp == 'uniform':
        amplitude = np.ones_like(amplitude)
    elif Amp == 'log':
        min_amplitude = np.min(amplitude[amplitude > 0])
        amplitude = np.log(amplitude / min_amplitude) / np.log(amplitude.max())
    elif Amp != 'raw':
        raise ValueError("Amp must be 'uniform', 'log', or 'raw'.")

    # Create RGB image
    rgb_image = np.zeros((*amplitude.shape, 3))
    rgb_image[..., 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
    rgb_image[..., 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amplitude  # Green
    rgb_image[..., 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

    if not subplot:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(rgb_image)
        plt.show()
    
    if scale:
        # Color wheel
        phase = np.linspace(0, 2 * np.pi, 256)
        r = 0.5 * (np.sin(phase) + 1)
        g = 0.5 * (np.sin(phase + np.pi / 2) + 1)
        b = 0.5 * (-np.sin(phase) + 1)
        colorwheel = np.stack([r, g, b], axis=1)
        warphase = ListedColormap(colorwheel)

        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        z = x + 1j * y
        mask = x**2 + y**2 > 1
        z[mask] = np.nan

        hue = np.angle(z)     
        idx = np.clip(((hue + np.pi) / (2 * np.pi) * 255).astype(int), 0, 255)
        color_wheel = colorwheel[idx]
        color_wheel = (color_wheel.T * np.abs(z)).T
        color_wheel = np.rot90(color_wheel, 2)
        color_wheel[np.isnan(color_wheel)] = 1.0
            
        if subplot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Unified figure creation
            axs[0].imshow(rgb_image)
            axs[0].axis('off')
            axs[1].imshow(color_wheel, origin='lower')
            axs[1].axis('off')
            sm = ScalarMappable(cmap=warphase)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[1], orientation='vertical', ticks=[0, 0.25, 0.5, 0.75, 1], fraction=0.046, pad=0.04)
            cbar.set_label('Hue', rotation=270, labelpad=15)
            cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(color_wheel, origin='lower')
            plt.axis('off')
            plt.show()


    return rgb_image

#TODO: assert that unfold_domain is a string and is included in a list of possible unfolding methods
#TODO: add other method of unfolding bsed on patterns (forgot the name)
def unfold_tensor(tensor, unfold_domain, undo=False, original_shape=None):
    
    
    if original_shape and unfold_domain in ['real', 'reciprocal', 'both']:
        tensor = tensor.reshape(original_shape)
        return tensor
    
    if not undo:
        original_shape = tensor.shape
    
    # Unfold real space and obtain a stack of diffraction patterns
    if unfold_domain == 'real':
        
        new_tensor_shape = (original_shape[0]*original_shape[1],) + original_shape[2:]
        tensor = tensor.reshape(new_tensor_shape)
    
    # Unfold reciprocal space and obtain a stack of real-space images
    elif unfold_domain == 'reciprocal':
        new_tensor_shape = original_shape[:2] + (original_shape[2]*original_shape[3],)
        tensor = tensor.reshape(new_tensor_shape)
    
    # Create a large 2D array with mixed coordinates
    elif unfold_domain == 'both':
        new_tensor_shape = (original_shape[0]*original_shape[1],) + (original_shape[2]*original_shape[3],)
        tensor = tensor.reshape(new_tensor_shape)
    
    elif unfold_domain == 'coordinate_aligned_real':
               
        if not undo:
            # From (0, 1, 2, 3, ...) we swap 1 and 2
            tensor = np.transpose(tensor, (0, 2, 1, 3, ))
            
            new_tensor_shape = (original_shape[0]*original_shape[2],) + (original_shape[1]*original_shape[3],)
            tensor = tensor.reshape(new_tensor_shape) 
        
        elif undo:
            
            intermediate_shape = (original_shape[0], original_shape[2], original_shape[1], original_shape[3],)
            tensor = tensor.reshape(intermediate_shape)
            # print(tensor)
            tensor = np.transpose(tensor, (0, 2, 1, 3, ))
    
    elif unfold_domain == 'coordinate_aligned_reciprocal':
               
        if not undo:
            # From (0, 1, 2, 3, ...) we swap 1 and 2
            tensor = np.transpose(tensor, (2, 0, 3, 1 ))
            
            new_tensor_shape = (original_shape[2]*original_shape[0],) + (original_shape[3]*original_shape[1],)
            tensor = tensor.reshape(new_tensor_shape) 
        
        elif undo:
            
            intermediate_shape = (original_shape[2], original_shape[0], original_shape[3], original_shape[1],)
            tensor = tensor.reshape(intermediate_shape)
            # print(tensor)
            tensor = np.transpose(tensor, (1, 3, 0, 2))
    
    # Create a typical 
    # elif unfold_domain == 'coordinate_aligned_real':
    #     new_tensor_shape = (original_shape[0]*original_shape[1],) + (original_shape[2]*original_shape[3],)
    #     tensor = tensor.reshape(new_tensor_shape) 
    
    # By default, the serpentine method unfolds real-space
    elif unfold_domain == 'serpentine':
        #TODO add serpentine unfolding
        pass
    
    elif unfold_domain == 'spiral_real':
        #TODO add spiral unfolding
        
        if not undo:
            pass
        
        elif undo:
            pass
    
    elif unfold_domain == 'spiral_reciprocal':
        #TODO add spiral unfolding
        pass
    
    return tensor


def clip_values(array, a_min=1, a_max=None):
    """Clip values in a numpy array to a specified range.

    This function clips the values in the input array to lie within the specified minimum and maximum 
    limits. Values less than `a_min` are set to `a_min`, and values greater than `a_max` (if provided) 
    are set to `a_max`. If `a_max` is not specified, no upper limit is applied.

    Parameters
    ----------
    array : np.ndarray
        The input numpy array whose values are to be clipped.
    a_min : float or int, optional
        The minimum value to which the elements in the array are clipped. Default is 1.
    a_max : float or int, optional
        The maximum value to which the elements in the array are clipped. If not specified, no upper 
        limit is applied.

    Returns
    -------
    clipped_array : np.ndarray
        The numpy array with values clipped to the specified range.

    Examples
    --------
    >>> arr = np.array([0, 2, 5, 10])
    >>> clip_values(arr, a_min=1, a_max=5)
    array([1, 2, 5, 5])

    Notes
    -----
    This function is a wrapper around `numpy.clip`, which performs the actual clipping operation. 
    The `a_min` and `a_max` parameters are inclusive, meaning that any value equal to `a_min` or 
    `a_max` will remain unchanged.
    """
    return np.clip(array, a_min, a_max)


def plot_centers_of_mass_with_histograms(centers_data, colors, labels=None, drawConvexHull=True,
                                         transparency=0.5, hist_height=0.16, bins=150,
                                         alpha=0.5, density=True, label_size=22, tick_label_size=18,
                                         x_range=(-0.9, 0.9), y_range=(-0.9, 0.9), hist_title_size=16):
    """
    Plots the centers of mass for all features from each dataset, centered at (0, 0),
    with histograms of the x and y coordinates.

    centers_data: The dataset with shape (n_datasets, A1, A2, n_spots, 2)
    hist_height: Height of the histograms as a fraction of total figure height
    label_size: Font size for the labels
    """

    # Check data compatibility
    assert type(drawConvexHull) is bool, "'drawConvexHull' must be a boolean (True/False) variable"
    assert type(density) is bool, "'density' must be a boolean  (True/False) variable"
    assert len(colors) == centers_data.shape[0], "The number of colors must match the number of datasets to plot."
    assert all(isinstance(item, str) for item in colors), "Not all elements are strings."

    n_datasets = centers_data.shape[0]

    # Create the main plot
    fig = plt.figure(figsize=(9, 9))
    ax_scatter = plt.axes([0.1, 0.1, 0.65, 0.65])
    ax_histx = plt.axes([0.1, 0.75, 0.65, hist_height], sharex=ax_scatter)
    ax_histy = plt.axes([0.75, 0.1, hist_height, 0.65], sharey=ax_scatter)

    # Disable labels on histogram to prevent overlap
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    # Initialize standard deviation lists
    std_dev_y = []
    std_dev_x = []

    markers = ['*', 'D', 's', '.', 'v', 'o', 'P', 'X']  # Different markers
    line_styles = ['--', '-', '-.', ':']  # Different line styles

    for i in range(n_datasets):
        all_x_coords = []
        all_y_coords = []

        # Collect all coordinates from all feature indices for the current dataset
        for feature_index in range(centers_data.shape[3]):
            y_coords = centers_data[i, :, :, feature_index, 0].flatten()
            x_coords = centers_data[i, :, :, feature_index, 1].flatten()
            y_mean, x_mean = np.mean(y_coords), np.mean(x_coords)

            all_y_coords.extend(y_coords - y_mean)
            all_x_coords.extend(x_coords - x_mean)

            # Scatter plot for each feature of the dataset with different markers
            if feature_index == 1:
                transparency /= 3
            ax_scatter.scatter(x_coords - x_mean, y_coords - y_mean, color=colors[i], alpha=transparency,
                               marker=markers[i % len(markers)])

        # Calculate and store standard deviations
        std_dev_y.append(np.std(all_y_coords))
        std_dev_x.append(np.std(all_x_coords))

        # Combine all x and y coordinates
        combined_coords = np.column_stack((all_x_coords, all_y_coords))

        # Draw convex hull for the combined coordinates of the dataset with different line styles
        if drawConvexHull and len(combined_coords) > 2:
            hull = ConvexHull(combined_coords)
            for simplex in hull.simplices:
                ax_scatter.plot(combined_coords[simplex, 0], combined_coords[simplex, 1], color=colors[i],
                                linewidth=2, linestyle=line_styles[i % len(line_styles)])

        # Add label for the dataset
        if labels is not None:
            ax_scatter.plot([], [], color=colors[i], label=labels[i], linestyle='None',
                            marker=markers[i % len(markers)], markerfacecolor=colors[i])
        else:
            ax_scatter.plot([], [], color=colors[i], label=f'Dataset {i+1}', linestyle='None',
                            marker=markers[i % len(markers)], markerfacecolor=colors[i],)

        # Plot histograms
        ax_histx.hist(all_x_coords, bins=bins, color=colors[i], alpha=alpha,
                      density=density, label=rf'$\sigma$ = {std_dev_x[-1]:.2f}')
        ax_histy.hist(all_y_coords, bins=bins, color=colors[i], alpha=alpha, orientation='horizontal',
                      density=density, label=rf'$\sigma$ = {std_dev_y[-1]:.2f}')

        print(rf'$\sigma$ = {std_dev_x[-1]:.2f}')
        print(rf'$\sigma$ = {std_dev_y[-1]:.2f}')

    # Set labels and title for the scatter plot
    ax_scatter.set_xlabel(r'$k_x$ Displacement (px.)', fontsize=label_size)
    ax_scatter.set_ylabel(r'$k_y$ Displacement (px.)', fontsize=label_size)
    # ax_scatter.set_title('Centers of Mass with Histograms', fontsize=label_size)
    # ax_scatter.legend(fontsize=label_size)
    
    ax_scatter.set_xlim(x_range)
    ax_scatter.set_ylim(y_range)
    ax_scatter.set_xticks(np.linspace(-.8,.8,9))
    ax_scatter.set_yticks(np.linspace(-.8,.8,9))

    
    ax_scatter.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histx.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histy.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histx.set_title('Center of Mass Precision', fontsize=25, pad=30)

    # Set the tick labels for the histograms
    ax_histx.set_yticks([0, 1, 2, 3])
    ax_histy.set_xticks([0, 1, 2, 3])
    
    # ax_histx.set_title('Probability\n Density', fontsize=hist_title_size, rotation=270,pad=50)
    ax_histy.set_title('Noisy Dataset\n \n Denoised Dataset', fontsize=hist_title_size,pad=50)
    
    plt.show()

def plotStrain(strain_data, title='Strain', axis='on', lim_val=0.05, cmap='RdBu', **kwargs):
    """ Strain/Rotation map plotting """

    plt.figure(figsize=(4.5, 10))

    im1 = plt.imshow(strain_data, vmin=-lim_val, vmax=lim_val, cmap=cmap, **kwargs)

    # plt.xticks(np.arange(0, 255, 10))
    # plt.yticks(np.arange(0, 110, 10))
    # plt.yticks(np.arange(0, 60, step=10), size = 20)
    
    plt.axis(axis)
    plt.title(title)
    # plt.grid()
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = plt.colorbar(im1, cax=cax)
    cb.ax.tick_params(labelsize=10)

    plt.show()

def plotHist_andClusters(data, clusters, cluster_indices, bins=100, xrange=None, yrange=None, 
                         axis_title_size=18, tick_label_size=14, color='blue', outline_color='red'):
    """
    Plot a histogram from "data" and outline histograms for specified clusters.

    :param data: Data to be plotted.
    :param clusters: Array of integers indicating cluster membership for each data point.
    :param cluster_indices: List of cluster indices for which to plot histogram outlines.
    :param bins: Number of bins in the histogram.
    :param xrange: Tuple specifying the (min, max) range of the x-axis.
    :param color: Color of the main histogram.
    :param outline_color: Color of the outlines for specified clusters.
    """

    # Flatten data
    flattened_data = data.flatten()

    plt.figure(figsize=(6.5, 5))

    # Plot the main histogram
    counts, bin_edges, _ = plt.hist(
        flattened_data, bins=bins, range=xrange, color=color, alpha=0.5, label='All Data')

    # Calculate bin centers from edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    for index in cluster_indices:
        # Extract data for the current cluster
        cluster_data = flattened_data[clusters.flatten() == index]

        # Calculate histogram for the current cluster
        cluster_counts, _ = np.histogram(cluster_data, bins=bin_edges)

        # Plot histogram outline for the current cluster with the user-specified outline color
        plt.plot(bin_centers, cluster_counts,
                 label=f'Cluster {index}', color=outline_color, drawstyle='steps-mid')
    
    # Set the x-axis and y-axis range if specified
    if xrange:
        plt.xlim(xrange)
    if yrange:
        plt.ylim(yrange)
    
    for tick in plt.gca().get_xticks():
        plt.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
    
    # Set tick label sizes
    plt.xticks(fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)
    
    plt.xlabel('Strain (%)', fontsize=axis_title_size)
    plt.ylabel('Frequency', fontsize=axis_title_size)
    # plt.legend()
    plt.show()

def inpaint_diffraction(image, centers, radius=6):
    """
    Use biharmonic equations to interpolate gap regions of a 2D array.

    Parameters
    ----------
    image : ndarray
        Input image.
    centers : list of tuples
        List of center coordinates for the circles to mask.
    radius : int, optional
        Radius of circular regions to mask.

    Returns
    -------
    predicted_background : ndarray
        Image with masked pixels inpainted.
    """
    from skimage.restoration import inpaint_biharmonic
    
    def create_circular_mask(h, w, centers, radius):
        """
        This function creates a circular mask for a 2D array.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        for center_idx, center in enumerate(centers):
            
            y, x = center
            
            if type(radius) == np.ndarray:
                radius = radius[center_idx]
                
            for i in range(h):
                for j in range(w):
                    if np.sqrt((i - y)**2 + (j - x)**2) <= radius:
                        mask[i, j] = 1
        return mask

    # Generate mask for the image
    masked_image = image.copy()
    h, w = image.shape[:2]
    mask = create_circular_mask(h, w, centers, radius)

    # Apply biharmonic inpainting
    inpainted_image = inpaint_biharmonic(masked_image, mask, split_into_regions=False)

    return inpainted_image

def sort_peaks(peak_centers, center, order_length=None):
    """
    Sorts peak centers by distance from a center point and optionally resorts 
    them by angle in groups.

    Parameters
    ----------
    peak_centers : ndarray
        Array of shape (A, 2) containing the coordinates of the peak centers.
    center : tuple
        Tuple (center_y, center_x) specifying the center point for distance calculation.
    order_length : int, optional
        The number of elements in each group to sort based on angle after initial distance sorting.
        If None, no secondary sorting is performed.

    Returns
    -------
    sorted_array : ndarray
        Array of peak centers sorted by distance and optionally resorted by angle in groups.
    """
    center_y, center_x = center

    # Calculate distances from the center point
    distances = np.sqrt((peak_centers[:, 0] - center_y) ** 2 + (peak_centers[:, 1] - center_x) ** 2)
    sorted_indices = np.argsort(distances)
    sorted_peaks = peak_centers[sorted_indices]

    if order_length is not None and order_length > 1:
        # Secondary sorting by angle within groups defined by order_length
        num_full_groups = len(sorted_peaks) // order_length
        resorted_array = []

        for i in range(num_full_groups):
            start_idx = i * order_length
            end_idx = start_idx + order_length
            subgroup = sorted_peaks[start_idx:end_idx]
            angles = np.arctan2(subgroup[:, 1] - center_x, subgroup[:, 0] - center_y)
            angle_indices = np.argsort(angles)
            resorted_array.append(subgroup[angle_indices])

        # Process the remaining elements
        if len(sorted_peaks) % order_length != 0:
            start_idx = num_full_groups * order_length
            remaining_group = sorted_peaks[start_idx:]
            remaining_angles = np.arctan2(remaining_group[:, 1] - center_x, remaining_group[:, 0] - center_y)
            remaining_angle_indices = np.argsort(remaining_angles)
            resorted_array.append(remaining_group[remaining_angle_indices])

        # Concatenate all the groups back into a single array
        sorted_peaks = np.concatenate(resorted_array, axis=0)

    return sorted_peaks

def reconstruct_height(xGrad, yGrad, y_bds_flat, x_bds_flat, iterations=10, threshold_percent=0.5, 
                       returnGradients=True, max_window_size=7):
    
    """
    Make 3D reconstruction based on xGrad and yGrad information. For each 
    iteration, the gradient sign (without changing its magnitude) is refined 
    for surface continuity.
    """
    
    # Use the adaptive median filter for initial sign correction
    xCorr = fix_sign_errors_adaptive(xGrad, max_window_size=max_window_size, initial_window_size=3)
    yCorr = fix_sign_errors_adaptive(yGrad, max_window_size=max_window_size, initial_window_size=3)

    initial_threshold_percent = threshold_percent
    for i in tqdm(range(iterations), desc="Reconstructing height"):
        current_threshold_percent = initial_threshold_percent * (iterations - i) / iterations
               
        h_map = reconFromGradDir(yCorr, xCorr, plot=False)
        hmap_fixed = fix_tilt_and_height(h_map, y_bds_flat, x_bds_flat)

        if i < iterations - 1:
            
            # Obtain the gradients from the reconstructed map
            xGrad_recon, yGrad_recon = compute_gradients(hmap_fixed)

            # Compare the original, modified gradients xCorr, yCorr with xGrad_recon, yGrad_recon
            xDiff = np.abs(xCorr - xGrad_recon) 
            yDiff = np.abs(yCorr - yGrad_recon)

            # Determine threshold based on percentile
            xThreshold = np.percentile(xDiff, 100 - current_threshold_percent)
            yThreshold = np.percentile(yDiff, 100 - current_threshold_percent)
            
            # Flip signs where the difference exceeds the threshold
            xCorr[xDiff > xThreshold] *= -1
            yCorr[yDiff > yThreshold] *= -1
            
            # Fix signs of gradients using the adaptive function
            xCorr = fix_sign_errors_adaptive(xCorr, max_window_size=max_window_size, initial_window_size=3)
            yCorr = fix_sign_errors_adaptive(yCorr, max_window_size=max_window_size, initial_window_size=3)
                
            # Flip signs where the difference exceeds the threshold
            xCorr[xDiff > xThreshold] = 0.5 * (xCorr[xDiff > xThreshold] + xGrad_recon[xDiff > xThreshold])
            yCorr[yDiff > yThreshold] = 0.5 * (yCorr[yDiff > yThreshold] + yGrad_recon[yDiff > yThreshold])
                               
    if returnGradients:
        return hmap_fixed, xCorr, yCorr
        
    else: 
        
        return hmap_fixed


def fix_sign_errors_adaptive(img, max_window_size=7, initial_window_size=3, variance_threshold=0.05):
    
    # Ensure both dimensions of the initial window size are odd for center pixel calculation
    if initial_window_size % 2 == 0 or max_window_size % 2 == 0:
        raise ValueError("Both initial_window_size and max_window_size must be odd.")

    # Create a copy of the original image to store the corrected values.
    corrected_img = np.copy(img)

    # Get the dimensions of the image.
    rows, cols = img.shape

    # Slide through the image using the adaptive window.
    for i in range(rows):
        for j in range(cols):
            # Start with the initial window size
            window_size = initial_window_size
            margin = window_size // 2

            # Get the center pixel value
            center_val = img[i, j]

            # Loop to adaptively increase window size
            while window_size <= max_window_size:
                # Ensure the window does not exceed the image boundaries
                r_start = max(0, i - margin)
                r_end = min(rows, i + margin + 1)
                c_start = max(0, j - margin)
                c_end = min(cols, j + margin + 1)

                # Extract the current neighborhood window
                window = img[r_start:r_end, c_start:c_end]
                
                # Compute the median and variance of the current window
                window_median = np.median(window)
                window_variance = np.var(window)

                # Check if the variance is below a threshold, meaning it's not noisy
                if window_variance < variance_threshold:
                    break  # Stop increasing window size, variance is low
                
                # Otherwise, increase the window size and try again
                window_size += 2  # Increase the window size by 2 (to maintain odd size)
                margin = window_size // 2

            # Check the sign of the center pixel and the median
            if np.sign(window_median) != np.sign(center_val):
                corrected_img[i, j] *= -1

    return corrected_img


def reconFromGradDir(grad_x, grad_y, im_height=None, plot=True):
    
    """
    Function based on MATLAB code written by Colin Ophus (MM/YYYY) 
    """
    
    padding = 2  # padding of reconstruction space
    qMin = 0.00  # min spatial frequency in 1/pixels
    qMax = 0.25  # max spatial frequency in 1/pixels
    num_iter = 50
    step_size = 0.99

    # Coordinates
    im_size = grad_x.shape
    N = tuple(s * padding for s in im_size)
    qxa, qya = makeFourierCoords(N, 1)
    q2a = qxa**2 + qya**2

    # Operators
    q2inv = np.reciprocal(q2a)
    q2inv[0, 0] = 0
    qFilt = np.exp(q2a / (-2 * qMax**2))
    if qMin > 0:
        qFilt = qFilt * (1 - np.exp(q2a / (-2 * qMin**2)))
    qxOp = (-1j / 4) * qxa * q2inv * qFilt
    qyOp = (-1j / 4) * qya * q2inv * qFilt

    # Normalize the gradients
    grad_x = grad_x - np.median(grad_x)
    grad_y = grad_y - np.median(grad_y)

    # Mask updates
    vx = np.arange(im_size[0])
    vy = np.arange(im_size[1])
    mask = np.zeros(N, dtype=bool)
    mask[np.ix_(vx, vy)] = True
    mask_inv = ~mask

    # Reconstruct height
    recon_height = np.zeros(N)
    for a0 in range(num_iter):
        grad_x_recon = (np.roll(recon_height, shift=-1, axis=0) -
                        np.roll(recon_height, shift=1, axis=0)) / 2
        grad_y_recon = (np.roll(recon_height, shift=-1, axis=1) -
                        np.roll(recon_height, shift=1, axis=1)) / 2

        # Difference and masking
        grad_x_recon[mask] = grad_x_recon[mask] - grad_x.ravel()
        grad_y_recon[mask] = grad_y_recon[mask] - grad_y.ravel()
        grad_x_recon[mask_inv] = 0
        grad_y_recon[mask_inv] = 0

        recon_update = np.fft.ifft2(
            np.fft.fft2(grad_x_recon) * qxOp +
            np.fft.fft2(grad_y_recon) * qyOp
        ).real

        recon_height -= step_size * recon_update

    # Crop
    recon_height = recon_height[np.ix_(vx, vy)]

    # Plotting
    if plot:
        plt.figure(11)
        plt.clf()
        ax = plt.gca()
        Ip1 = recon_height - np.median(recon_height)
        if im_height is None:
            im = ax.imshow(Ip1, cmap='turbo')
        else:
            Ip2 = im_height - np.median(im_height)
            im = ax.imshow(np.hstack((Ip1, Ip2)), cmap='turbo')

        plt.axis('equal')
        plt.axis('off')

        # Adjust the colorbar to have the same height as the image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

    return recon_height

def makeFourierCoords(size, spacing):
    """
    This function is based on MATLAB code written by Colin Ophus (MM/YYYY).
    """
    nx, ny = size
    qx = np.fft.fftfreq(nx, spacing)
    qy = np.fft.fftfreq(ny, spacing)
    qxa, qya = np.meshgrid(qx, qy, indexing='ij')
    return qxa, qya

def fix_tilt_and_height(height_map, y_bds, x_bds):
    """
    Helper function to perform linear regression and return the slope
    """
    def get_slope(ave_height):
        x = np.arange(ave_height.size)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, ave_height, rcond=None)[0]
        return slope

    # 1. Linear regression for x-axis and adjust tilt
    ave_height_x = np.mean(height_map, axis=0)
    slope_x = get_slope(ave_height_x)
    height_map = rotate_height(
        height_map, np.arctan(slope_x) * 180 / np.pi, axis='x')

    # 2. Linear regression for y-axis and adjust tilt
    ave_height_y = np.mean(height_map, axis=1)
    slope_y = get_slope(ave_height_y)
    height_map = rotate_height(
        height_map, np.arctan(slope_y) * 180 / np.pi, axis='y')

    # 3. Adjust the average height based on a chosen region
    height_map = height_map - np.mean(height_map[y_bds[0]:y_bds[1], x_bds[0]:x_bds[1]])

    return height_map

def compute_gradients(height_map):
    """
    Compute gradients given a 3D map
    """
    
    phi, theta = get_surface_tilt_and_direction(height_map, units='rad', show_results=False)
    
    grads = get_gradients(np.stack((phi, theta), axis=-1), rot_angle=180)

    return grads[...,1], grads[...,0]  

def rotate_height(height, angle, axis='x'):
    """Rotate the height map based on the specified axis and angle.

    Parameters:
    - height: 2D numpy array representing the height map.
    - angle: float, rotation angle in degrees.
    - axis: str, 'x' for x-axis rotation, 'y' for y-axis rotation.

    Returns:
    - 2D numpy array representing the rotated height map.
    """

    shape_array = np.shape(height)

    if axis == 'x':
        i_values = np.arange(shape_array[1])
        adjustment = i_values * np.tan(angle * np.pi / 180)
        height_rot = height[:, :shape_array[1]] - adjustment
    elif axis == 'y':
        i_values = np.arange(shape_array[0])
        adjustment = i_values * np.tan(angle * np.pi / 180)
        height_rot = height[:shape_array[0], :] - adjustment[:, np.newaxis]

    return height_rot

def low_std_mask_and_average(data, percentile=5):
    """
    Generates a mask for the X% of pixels with the lowest standard deviation along the N axis
    and returns a vector of length N by averaging those masked pixels.

    Parameters:
    - data: numpy array of shape (A, B, N) where N is the number of features
    - percentile: Percentage of pixels to be considered with the lowest standard deviation

    Returns:
    - avg_vector: numpy array of length N representing the averaged values across the masked pixels
    """

    # Calculate standard deviation along the last axis
    std_dev = np.std(data, axis=2)
    
    # Determine the threshold for the lowest X% standard deviation
    threshold = np.percentile(std_dev, percentile)

    # Generate the mask where std_dev is less than or equal to the threshold
    mask = std_dev <= threshold

    # Plot the mask with flat pixels
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Flat Mask ({percentile}%)')
    plt.axis('off')
    plt.show()

    # Apply the mask and calculate mean intensities
    masked_data = data[mask, :]
    avg_vector = np.mean(masked_data, axis=0)

    return avg_vector

def mask_and_average(data, function='std', percentile=5, threshold='lower', 
                     return_mask=False, show_mask=True):
    """
    Generates a mask for the X% of pixels based on the specified metric along the last axis
    and returns a vector of length N (or N, M) by averaging those masked pixels.

    Parameters:
    - data: numpy array of shape (A, B, N) or (A, B, N, M)
    - function: Metric to calculate ('std_1d', 'sum_2d', etc.)
    - percentile: Percentage of pixels to be considered based on the metric
    - threshold: 'lower' or 'upper' to specify the percentile direction

    Returns:
    - avg_vector: numpy array representing the averaged values across the masked pixels
    """

    # Calculate the specified metric along the last axis
    metric = _calculate_metric(data, function)

    # Determine the threshold based on the specified percentile and direction
    if threshold == 'lower':
        thresh_value = np.percentile(metric, percentile)
        mask = metric <= thresh_value
    elif threshold == 'upper':
        thresh_value = np.percentile(metric, 100 - percentile)
        mask = metric >= thresh_value
    else:
        raise ValueError(f"Unsupported threshold type: {threshold}")
        
    if show_mask:
        # Plot the mask
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask ({threshold} {percentile}%) based on {function}')
        plt.axis('off')
        plt.show()

    # Apply the mask and calculate the mean across the masked pixels
    masked_data = data[mask]
    avg = np.mean(masked_data, axis=0)
    
    if return_mask:
        return avg, mask
    
    else:
        return avg

def _calculate_metric(data, function):
    """
    Calculate the metric based on the specified function.

    Parameters:
    - data: numpy array of shape (A, B, N) or (A, B, N, M)
    - function: String specifying the metric to calculate ('std', 'sum', etc.)

    Returns:
    - metric: 2D numpy array of shape (A, B) containing the calculated metric
    """
    if function == 'std_1d':
        metric = np.std(data, axis=-1)
    elif function == 'sum_2d':
        metric = np.sum(data, axis=(-2, -1))
    else:
        raise ValueError(f"Unsupported function: {function}")
    return metric

def get_average_clusters(dataset, cluster_map, 
                         plot_averages=True, vmin=4, vmax=14,cmap='turbo',logScale=False):
    
    A, B, C, D = dataset.shape
    E, F = cluster_map.shape
    
    assert A == E and B == F, "The 1st and 2nd dimensions of 'dataset' must match the dimensions of 'cluster_map'" 
    
    # Count number of clusters
    n_clusters = len(np.unique(cluster_map))
    average_values = np.zeros((n_clusters, C, D))
    
    # Loop over each cluster
    for cluster_idx in range(n_clusters):
            
        # Calculate the average across all selected patterns for this cluster
        average_values[cluster_idx] = np.mean(dataset.array[cluster_map == cluster_idx], axis=0)
        
    if plot_averages:
        for i in range(n_clusters):
            if logScale:
                plt.imshow(np.log(average_values[i]+1), vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                plt.imshow(average_values[i]+1,vmin=vmin,vmax=vmax,cmap=cmap)
            plt.axis(False)
            plt.show()
    
    return HyperData(average_values)

def get_cluster_masks(clusterMap):
    """
    Separate the clusters in cluster map
    """
    
    A, B = clusterMap.shape
    labels = np.unique(clusterMap)

    cluster_masks = np.zeros((len(labels),A,B), dtype=bool)
    
    for label in labels:
        
        cluster_masks[label][clusterMap == label] = clusterMap[clusterMap == label]
        
def shift_mask_integer(mask: np.ndarray, 
                       dy: int = 0, 
                       dx: int = 0) -> np.ndarray:
    """
    Shift a boolean mask by integer dy, dx with padding of False (no wrap).

    Parameters
    ----------
    mask : np.ndarray of bool
    dy : int
        Shift in the y–direction (positive moves content *down*).
    dx : int
        Shift in the x–direction (positive moves content *right*).

    Returns
    -------
    shifted : np.ndarray of bool
    """
    h, w = mask.shape
    # create empty output
    shifted = np.zeros_like(mask, dtype=bool)

    # source and destination slicing
    # dest y-range
    y0_dst = max(dy, 0)
    y1_dst = min(h + dy, h)
    # dest x-range
    x0_dst = max(dx, 0)
    x1_dst = min(w + dx, w)

    # corresponding source ranges
    y0_src = max(-dy, 0)
    y1_src = min(h - dy, h)
    x0_src = max(-dx, 0)
    x1_src = min(w - dx, w)

    shifted[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return shifted

def fill_nans_1d(arr: np.ndarray) -> np.ndarray:
    """
    Return a copy of the 1D array `arr` where NaNs have been replaced by
    linear interpolation between the surrounding valid points.
    
    Leading/trailing NaNs are filled with the first/last valid value.
    
    Parameters
    ----------
    arr : np.ndarray, shape (N,)
        Input array containing floats and possibly NaNs.
    
    Returns
    -------
    out : np.ndarray, shape (N,)
        Copy of `arr` with NaNs replaced by interpolated values.
    
    Raises
    ------
    ValueError
        If `arr` is not 1D or if all values are NaN.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported")
    
    x = np.arange(arr.size)
    mask_good = ~np.isnan(arr)
    
    if not mask_good.any():
        raise ValueError("Array contains only NaNs; cannot interpolate")
    
    # np.interp: for x < first_good it returns arr[first_good],
    # for x > last_good returns arr[last_good].
    filled = arr.copy()
    filled[np.isnan(arr)] = np.interp(
        x[np.isnan(arr)],
        x[mask_good],
        arr[mask_good]
    )
    return filled

def merge_cluster_maps(cluster_maps):
    """
    Merge N cluster maps (same shape) by fragmenting overlapping regions.
    Each pixel has an N‐tuple of original labels; we assign a new unique
    label to each distinct N‐tuple, and remap background (all zeros) → 0.

    Parameters
    ----------
    cluster_maps : list of np.ndarray, each shape (H, W), integer labels
        The input cluster maps to merge. 0 is background.

    Returns
    -------
    merged_map : np.ndarray[int], shape (H, W)
        Integer map where each unique combination of input‐map labels
        has its own cluster ID (0 for background).
    """
    # stack into (H, W, N) and flatten to (H*W, N)
    stack = np.stack(cluster_maps, axis=-1)
    H, W, N = stack.shape
    flat = stack.reshape(-1, N)

    # build map from tuple of original labels → new ID
    combo_to_id = {}
    next_id = 1
    merged_flat = np.zeros(flat.shape[0], dtype=int)

    for i, row in enumerate(flat):
        key = tuple(row)
        if key not in combo_to_id:
            combo_to_id[key] = next_id
            next_id += 1
        merged_flat[i] = combo_to_id[key]

    # background key is all zeros
    bg_key = tuple([0]*N)
    if bg_key in combo_to_id:
        bg_id = combo_to_id[bg_key]
        # set background pixels to 0
        merged_flat[merged_flat == bg_id] = 0
        # shift down any IDs > bg_id by 1 so labels remain contiguous
        merged_flat = merged_flat - (merged_flat > bg_id).astype(int)

    # reshape back to (H, W)
    merged_map = merged_flat.reshape(H, W)
    return merged_map

from scipy.ndimage import binary_dilation

def filter_and_split_cluster_map(cluster_map: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Split each integer-labeled region into connected-component "islands",
    reassign any island smaller than min_size to the majority neighboring island,
    and renumber islands so labels run from 0..M-1 consecutively.

    Notes
    -----
    - There is no special "background" label; every integer label in `cluster_map`
      is treated the same (including 0, if present).
    - Connected components use 4-connectivity (same default as scipy.ndimage.label).
    - Neighbor voting uses an 8-neighborhood around the island boundary.

    Parameters
    ----------
    cluster_map : np.ndarray[int], shape (H, W)
        Input 2D array of cluster labels (all labels treated equally).
    min_size : int
        Minimum island size to keep. Islands smaller than this are reassigned
        to the most common neighboring island label when possible.

    Returns
    -------
    np.ndarray[int], shape (H, W)
        Filtered and fully-split cluster map, with new labels 0..M-1.
    """
    cluster_map = np.asarray(cluster_map)
    H, W = cluster_map.shape

    comp_global = np.zeros((H, W), dtype=int)
    next_id = 0
    island_sizes: dict[int, int] = {}

    # Step 1: split every label (including 0) into connected-component islands
    for lab in np.unique(cluster_map):
        mask_lab = (cluster_map == lab)
        if not mask_lab.any():
            continue
        comp_lab, num_comp = label(mask_lab)  # default 4-connectivity
        for comp_idx in range(1, num_comp + 1):
            comp_mask = (comp_lab == comp_idx)
            comp_size = int(comp_mask.sum())
            comp_global[comp_mask] = next_id
            island_sizes[next_id] = comp_size
            next_id += 1

    # Step 2: reassign small islands based on majority neighbor (no background special-casing)
    neigh_struct = np.ones((3, 3), dtype=bool)  # 8-neighborhood for boundary expansion

    for island_id in sorted(island_sizes, key=island_sizes.get):
        if island_sizes[island_id] >= min_size:
            continue
        if not np.any(comp_global == island_id):  # may have been merged already
            continue

        mask = (comp_global == island_id)
        border = binary_dilation(mask, structure=neigh_struct) & ~mask
        neigh = comp_global[border]
        neigh = neigh[neigh != island_id]

        if neigh.size:
            target = int(np.bincount(neigh).argmax())
            comp_global[mask] = target
        # else: no neighbors (e.g., single-island map); keep it as-is

    # Step 3: renumber islands consecutively (0..M-1)
    old_ids = np.unique(comp_global)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    new_map = np.vectorize(remap.get, otypes=[int])(comp_global).astype(int)

    return new_map

#%%

def spiral_matrix(matrix, return_indices=True):
    """
    Extract elements from a 2D NumPy array in a spiral order or returns the flat indices of the spiral order.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D array from which to extract elements in a spiral order.
    indices : bool, optional
        If True, returns the flat indices of the elements in spiral order.
        If False, returns the elements themselves.
    
    Returns
    -------
    numpy.ndarray
        An array of values or indices in spiral order.
    
    Examples
    --------
    >>> matrix = np.array([[10, 20], [30, 40]])
    >>> spiral_matrix(matrix, indices=False)
    array([10, 20, 40, 30])
    >>> spiral_matrix(matrix, indices=True)
    array([0, 1, 3, 2])
    
    Notes
    -----
    This function assumes the input matrix is 2D and at least 1x1 in size. The indices are computed relative to the flattened matrix.
    """
    
    
    result = []
    while matrix.size > 0:
        # Add the first row
        result.append(matrix[0, :])
        matrix = matrix[1:]  # Remove the first row
        
        if matrix.size == 0:
            break
        
        # Add the last column
        result.append(matrix[:, -1])
        matrix = matrix[:, :-1]  # Remove the last column
        
        if matrix.size == 0:
            break
        
        # Add the last row reversed
        result.append(matrix[-1, ::-1])
        matrix = matrix[:-1]  # Remove the last row
        
        if matrix.size == 0:
            break
        
        # Add the first column reversed
        result.append(matrix[::-1, 0])
        matrix = matrix[:, 1:]  # Remove the first column

    return np.concatenate(result)

# # Example usage
# matrix = np.linspace(0,7**2-1,7**2).reshape(7,7)
# # matrix = (np.random.random((5,5))*100).astype(int)
# spiral = spiral_matrix(matrix)
# print(spiral)
# # print(spiral)

def gradient_ascent(data, start, learning_rate=0.1, max_iters=100):
    y, x = start
    for i in range(max_iters):
        grad_y, grad_x = np.gradient(data)
        y += learning_rate * grad_y[int(y), int(x)]
        x += learning_rate * grad_x[int(y), int(x)]
        if grad_y[int(y), int(x)] == 0 and grad_x[int(y), int(x)] == 0:
            break
    return y, x

# Gaussian fitting
def gaussian_2d(xdata, y0, x0, yalpha, xalpha, amplitude, offset):
    y, x = xdata
    return offset + amplitude * np.exp(
        -(((y - y0) ** 2 / (2 * yalpha ** 2)) + ((x - x0) ** 2 / (2 * xalpha ** 2)))
    )

def fit_gaussian_2d(data):
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((y.ravel(), x.ravel()))
    initial_guess = (data.shape[0]//2, data.shape[1]//2, 1, 1, data.max(), 0)
    popt, _ = curve_fit(gaussian_2d, xdata, data.ravel(), p0=initial_guess)
    return popt[0], popt[1]

# Elliptical Gaussian fitting
def elliptical_gaussian_2d(xdata, y0, x0, yalpha, xalpha, theta, amplitude, offset):
    y, x = xdata
    a = (np.cos(theta)**2 / (2 * xalpha**2)) + (np.sin(theta)**2 / (2 * yalpha**2))
    b = -(np.sin(2*theta) / (4 * xalpha**2)) + (np.sin(2*theta) / (4 * yalpha**2))
    c = (np.sin(theta)**2 / (2 * xalpha**2)) + (np.cos(theta)**2 / (2 * yalpha**2))
    return offset + amplitude * np.exp(-(a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2)))

def fit_elliptical_gaussian_2d(data):
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((y.ravel(), x.ravel()))
    initial_guess = (data.shape[0]//2, data.shape[1]//2, 1, 1, 0, data.max(), 0)
    popt, _ = curve_fit(elliptical_gaussian_2d, xdata, data.ravel(), p0=initial_guess)
    return popt[0], popt[1]

def mask_corrupted_pixels(arr, threshold=3, window_size=3):
    """
    Identify and mask corrupted pixels in a 2D array based on a threshold difference
    from the neighboring area, which is defined by the window size.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.
    threshold : float, optional
        The threshold difference to identify corrupted pixels. A pixel is considered
        corrupted if its value differs from the mean of its neighbors by more than
        this threshold. Default is 3.
    window_size : int, optional
        The size of the window used to define the neighboring area around each pixel.
        Must be an odd integer. Default is 3.

    Returns
    -------
    mask : np.ndarray
        A 2D boolean array where `True` indicates a corrupted pixel.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")
    
    # Initialize the mask with False
    mask = np.zeros_like(arr, dtype=bool)
    
    # Calculate the offset based on the window size
    offset = window_size // 2
    
    # Get the dimensions of the array
    rows, cols = arr.shape
    
    # Iterate over each pixel (excluding the border pixels)
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            # Extract the pixel's neighboring area based on the window size
            neighbors = arr[i-offset:i+offset+1, j-offset:j+offset+1]
            neighbors_mean = np.std(neighbors)
            
            # Calculate the difference between the pixel and the mean of its neighbors
            if np.abs(arr[i, j] - neighbors_mean) > threshold:
                mask[i, j] = True
                
    return mask

import numpy as np
import matplotlib.pyplot as plt

def get_gradients(sol_array, rot_angle=0, show_result=False, invertY=False, invertX=False, cbar_fraction=0.04):
    """Calculate directional gradients from a solution array of (phi, theta) values.

    This function computes the directional gradients (dz/dx and dz/dy) from a solution array that 
    contains azimuthal (phi) and elevation (theta) angles. The gradients are computed considering 
    a specified rotation angle. Optionally, the gradients can be visualized as 2D images.

    Parameters
    ----------
    sol_array : ndarray
        A 3D numpy array of shape (rows, cols, 2) where the last dimension represents the angles 
        (phi, theta) in radians.
    rot_angle : float, optional
        The rotation angle in degrees to adjust the azimuthal angle (phi). Default is 0.
    show_result : bool, optional
        If True, displays the gradients as images with colorbars. Default is False.
    invertY : bool, optional
        If True, inverts the y-axis gradient. Default is False.
    invertX : bool, optional
        If True, inverts the x-axis gradient. Default is False.

    Returns
    -------
    array_gradients : ndarray
        A 3D numpy array of shape (rows, cols, 2) containing the computed gradients. 
        The last dimension holds the gradients (dz/dx, dz/dy).

    Examples
    --------
    Given a solution array `sol_array` with shape (100, 100, 2):
    
    >>> gradients = get_gradients(sol_array, rot_angle=45, show_result=True)
    This will calculate the gradients after rotating phi by 45 degrees and display the results.

    Notes
    -----
    The function assumes that the input array contains azimuthal angles in the first channel 
    (phi) and elevation angles in the second channel (theta). The rotation angle is applied 
    to the azimuthal angle before computing the gradients. The option to invert the gradients 
    allows flexibility in adjusting the gradient directions based on the specific application.
    """

    # Adjust phi by the rotation angle
    phi = sol_array[:, :, 0] - rot_angle * np.pi / 180
    theta = sol_array[:, :, 1]
    
    # Compute gradients
    dzdy = np.tan(theta) * np.cos(phi)
    dzdx = np.tan(theta) * np.sin(phi)

    # Apply inversion if required
    if invertY:
        dzdy = -dzdy
    if invertX:
        dzdx = -dzdx

    # Stack gradients into a single array
    array_gradients = np.stack((dzdx, dzdy), axis=-1)
    
    if show_result:
        # Plotting the gradients
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Gradient X
        im1 = axs[0].imshow(array_gradients[:,:,0], cmap='RdBu')
        axs[0].axis('off')
        axs[0].set_title('Gradient X', fontsize=10)
        cbar1 = fig.colorbar(im1, ax=axs[0], fraction=cbar_fraction)
        cbar1.ax.set_title(r'$\nabla_X$', fontsize=8, pad=5)
        
        # Gradient Y
        im2 = axs[1].imshow(array_gradients[:,:,1], cmap='RdBu')
        axs[1].axis('off')
        axs[1].set_title('Gradient Y', fontsize=10)
        cbar2 = fig.colorbar(im2, ax=axs[1], fraction=cbar_fraction)
        cbar2.ax.set_title(r'$\nabla_Y$', fontsize=8, pad=5)
        
        plt.tight_layout()
        plt.show()
    
    return array_gradients

def correct_strain(exx, eyy, exy, phi, theta):
    """
    Correct strain arrays for in-plane surface tilt
    
    The input strain arrays `exx`, `eyy`, and `exy` must be corrected to account for 
    the in-plane tilt of the surface, described by the azimuthal (`phi`) and elevation (`theta`) angles.
    
    Parameters
    ----------
    exx : array
       Uniaxial strain array in the x-direction
    eyy : array
       Uniaxial strain array in the y-direction
    exy : array
       Shear strain array
    phi : array
       Array of azimuthal angles for each position's local tilt
    theta : array
       Array of elevation angles for each position's local tilt
    
    Returns
    -------
    corrected_exx : array
       Corrected uniaxial strain array in the x-direction
    corrected_eyy : array
       Corrected uniaxial strain array in the y-direction
    corrected_exy : array
       Corrected shear strain array
    
    Examples
    --------
    Correct strain arrays with given tilt angles:
    >>> corrected_exx, corrected_eyy, corrected_exy = correct_strain(exx, eyy, exy, phi, theta)
    Corrected arrays aligned to the tilt angles
    
    Notes
    -----
    This function applies Mohr's circle transformations to align with `phi` 
    and then corrects for the tilt in `theta`.
    """
                    
    # Apply Mohr's circle transformations
    e_parallel =       (exx + eyy)/2 + (exx - eyy)/2*np.cos(2*phi) + exy*np.sin(2*phi)
    e_perpendicular =  (exx + eyy)/2 - (exx - eyy)/2*np.cos(2*phi) - exy*np.sin(2*phi)
    e_shear =         -(exx - eyy)/2*np.sin(2*phi) + exy*np.cos(2*phi)
    
    # Apply strain correction using physical tilt information
    e_parallel = (e_parallel + 1)/np.cos(theta) - 1
    
    # Return to original orientation
    exx_corr =  (e_parallel + e_perpendicular)/2 + (e_parallel - e_perpendicular)/2*np.cos(-2*phi) + e_shear*np.sin(-2*phi)
    eyy_corr =  (e_parallel + e_perpendicular)/2 - (e_parallel - e_perpendicular)/2*np.cos(-2*phi) - e_shear*np.sin(-2*phi)
    exy_corr =  -(e_parallel - e_perpendicular)/2*np.sin(-2*phi) + e_shear*np.cos(-2*phi)

    return exx_corr, eyy_corr, exy_corr


def rotate_strain(exx, eyy, exy, alpha):
    """

    Parameters
    ----------
    exx : array
       Uniaxial strain array in the x-direction
    eyy : array
       Uniaxial strain array in the y-direction
    exy : array
       Shear strain array
    alpha : float
       Angle (radians)
    
    Returns
    -------
    e_parallel : array
       uniaxial strain array in the alpha-direction
    e_perpendicular : array
       uniaxial strain array in the direction perpendicular to alpha
    e_shear : array
       shear strain array
    
    Examples
    --------
    Correct strain arrays with given tilt angles:
    >>> corrected_exx, corrected_eyy, corrected_exy = correct_strain(exx, eyy, exy, phi, theta)
    Corrected arrays aligned to the tilt angles
    
    Notes
    -----
    This function applies Mohr's circle transformations to align with `alpha` 
    and then corrects for the tilt in `theta`.
    """
                    
    # Apply Mohr's circle transformations
    e_parallel =       (exx + eyy)/2 + (exx - eyy)/2*np.cos(2*alpha) + exy*np.sin(2*alpha)
    e_perpendicular =  (exx + eyy)/2 - (exx - eyy)/2*np.cos(2*alpha) - exy*np.sin(2*alpha)
    e_shear =         -(exx - eyy)/2*np.sin(2*alpha) + exy*np.cos(2*alpha)
    
    return e_parallel, e_perpendicular, e_shear       

def bin_array_with_padding(array, new_shape, padding=10):
    """
    Bin an array by averaging over the first two axes (ny, nx) 
    to resize it to the new shape (ny_new, nx_new), with white padding.
    
    Parameters:
        array (ndarray): Input array of shape (ny, nx, ky, kx).
        new_shape (tuple): Desired shape for the first two axes (ny_new, nx_new).
        padding (int): Number of pixels for the white padding between images.
        
    Returns:
        visualization_grid (ndarray): 2D array representing the padded grid visualization.
    """
    ny, nx, ky, kx = array.shape
    ny_new, nx_new = new_shape

    # Compute binning factors
    bin_size_y = ny / ny_new
    bin_size_x = nx / nx_new

    # Initialize the binned array
    binned_array = np.zeros((ny_new, nx_new, ky, kx))

    for i in range(ny_new):
        for j in range(nx_new):
            # Determine the slice indices for the bin
            y_start = int(i * bin_size_y)
            y_end = int((i + 1) * bin_size_y)
            x_start = int(j * bin_size_x)
            x_end = int((j + 1) * bin_size_x)

            # Take the average over the selected slice
            binned_array[i, j] = array[y_start:y_end, x_start:x_end].mean(axis=(0, 1))

    # Pad each image with white space
    padded_images = np.array(
        [
            [
                np.pad(
                    binned_array[i, j], 
                    pad_width=((padding, padding), (padding, padding)), 
                    mode="constant", 
                    constant_values=0
                )
                for j in range(nx_new)
            ]
            for i in range(ny_new)
        ]
    )

    # Combine the padded images into a grid
    visualization_grid = np.block([[padded_images[i, j] for j in range(nx_new)] for i in range(ny_new)])

    return visualization_grid


def rolling_ball_background(image, radius, smooth=True, smooth_sigma_frac=0.25):
    """
    Estimate a smooth 2D background using a rolling-ball (sphere) algorithm.

    Parameters
    ----------
    image : (H, W) array_like
        Input 2D image (e.g., diffraction pattern). Casts to float.
    radius : float
        Ball radius in pixels. Sets the length scale of the background.
    smooth : bool, optional
        If True, apply a small Gaussian blur to the background at the end.
    smooth_sigma_frac : float, optional
        Fraction of `radius` used as Gaussian sigma when `smooth=True`.

    Returns
    -------
    background : (H, W) ndarray
        Estimated background image.
    """
    im = np.asarray(image, dtype=float)

    # Build a spherical structuring element (ball) in 2D
    r = int(np.ceil(radius))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    dist2 = xx*xx + yy*yy
    mask = dist2 <= radius**2

    ball = np.zeros_like(dist2, dtype=float)
    ball[mask] = np.sqrt(radius**2 - dist2[mask])
    # Top of the ball at 0, rest <= 0 (what grey_* expects)
    ball -= ball.max()

    # Morphological opening with a spherical structuring element
    eroded = grey_erosion(im, footprint=mask, structure=ball)
    background = grey_dilation(eroded, footprint=mask, structure=ball)

    if smooth and radius > 0:
        sigma = smooth_sigma_frac * radius
        background = gaussian_filter(background, sigma=sigma)

    return background

def split_disconnected_clusters(cluster_map,
                                connectivity: int = 2,
                                background: int = -1):
    """
    Split disconnected "islands" within each cluster label into separate clusters.

    Parameters
    ----------
    cluster_map : (H, W) ndarray of int
        2D map of cluster labels. Pixels with the same integer label belong
        to the same cluster (before splitting).
    connectivity : {1, 2}, optional
        Connectivity for defining local adjacency:
        - 1 → 4-connected (up, down, left, right)
        - 2 → 8-connected (also includes diagonals)
    background : int, optional
        Label to treat as background (left unchanged and not split).

    Returns
    -------
    new_map : (H, W) ndarray of int
        Relabeled map where every spatially connected component that was part
        of a given label gets its own unique new label. New labels are
        consecutive integers starting from 0 (excluding background).
    mapping : dict
        Dictionary describing how old labels were split:
        keys:   (old_label, component_index) where component_index ∈ {1..n_islands}
        values: new_label (int) in `new_map`.

        Example:
            mapping[(3, 1)] = 0   # first island of old label 3 → new label 0
            mapping[(3, 2)] = 1   # second island of old label 3 → new label 1

    Notes
    -----
    - Two regions with the same original label but not touching (given the
      specified connectivity) will become distinct clusters in `new_map`.
    - The background label is preserved as-is wherever it appears.
    """

    cluster_map = np.asarray(cluster_map)
    if cluster_map.ndim != 2:
        raise ValueError("cluster_map must be a 2D array.")

    H, W = cluster_map.shape
    new_map = np.full((H, W), background, dtype=int)

    labels = np.unique(cluster_map)
    struct = ndimage.generate_binary_structure(2, connectivity)

    next_label = 0
    mapping = {}

    for lab in labels:
        if lab == background:
            # copy background pixels directly
            new_map[cluster_map == lab] = background
            continue

        mask = (cluster_map == lab)
        if not mask.any():
            continue

        # Label connected components within this mask
        comp_map, n_comp = ndimage.label(mask, structure=struct)

        for comp_id in range(1, n_comp + 1):
            comp_mask = (comp_map == comp_id)
            new_map[comp_mask] = next_label
            mapping[(lab, comp_id)] = next_label
            next_label += 1

    return new_map, mapping

def combine_strain_maps(strain_maps: np.ndarray,
                        all_intensities: np.ndarray,
                        peaks: np.ndarray,
                        n_fold: int = 6,
                        intensity_power: float = 1.0,
                        intensity_percentile: float | None = None,
                        min_total_intensity_frac: float = 0.0,
                        eps: float = 1e-12) -> np.ndarray:
    """
    Combine per-order strain maps into a final 4-component strain field using
    "smart" intensity weighting.

    Parameters
    ----------
    strain_maps : ndarray, shape (num_orders, 4, Ny, Nx)
        Strain/rotation maps per order (group of n_fold peaks).
        `strain_maps[o, s, y, x]` is the s-th strain component for order o
        at pixel (y, x). s = 0,1,2,3 → e_xx, e_yy, e_xy, rot (or similar).
    all_intensities : ndarray, shape (Ny, Nx, n_peaks)
        Bragg peak intensities for each peak at each pixel.
    peaks : ndarray, shape (n_peaks, 2)
        Peak coordinates (not used directly, only n_peaks is checked).
        Assumed grouped in contiguous blocks of size n_fold:
        [0..n_fold-1], [n_fold..2*n_fold-1], ...
    n_fold : int, optional
        Number of peaks per order (e.g. 6 for hexagonal, 4 for square).
    intensity_power : float, optional
        Exponent applied to intensities before weighting:
          - 1.0  → linear weighting by intensity
          - 0.5  → sub-linear (softens dominance of very bright peaks)
          - 2.0  → super-linear (emphasizes very bright peaks)
        Effective intensity is I_eff = I ** intensity_power.
    intensity_percentile : float in [0, 100] or None, optional
        If not None, intensities are clipped at this percentile of I_eff
        (over all peaks/pixels) before weighting. This prevents a few
        extreme pixels from dominating:
          vmax = percentile(|I_eff|, intensity_percentile)
          I_eff = clip(I_eff, 0, vmax)
    min_total_intensity_frac : float in [0, 1], optional
        Threshold for masking low-intensity pixels in the final map.
        Let T(x,y) = sum over all peaks of I_eff(x,y,peak). We compute
        T_max = T.max() and treat pixels with
            T(x,y) < min_total_intensity_frac * T_max
        as unreliable (weights too small). For those pixels, output is set
        to zero.
        Set = 0.0 to disable masking.
    eps : float, optional
        Small constant to avoid divide-by-zero.

    Returns
    -------
    final_strain_maps : ndarray, shape (4, Ny, Nx)
        Combined strain/rotation maps, intensity-weighted in a "smart" way.

    Notes
    -----
    The combination is:

        For each order o:
            group_weight_o(x,y) = sum_{k in group(o)} I_eff_k(x,y)

        Numerator_s(x,y) = sum_o [ strain_maps[o, s, x, y] * group_weight_o(x,y) ]
        Denominator(x,y) = sum_o group_weight_o(x,y)

        final_strain_maps[s, x, y] = Numerator_s(x,y) / Denominator(x,y)

    where I_eff_k is the possibly exponentiated and clipped intensity.
    """
    # ----------------- basic sanity checks ----------------- #
    Ny, Nx, n_peaks = all_intensities.shape
    num_orders = peaks.shape[0] // n_fold

    assert strain_maps.shape[0] == num_orders, (
        f"strain_maps.shape[0]={strain_maps.shape[0]} does not match "
        f"num_orders={num_orders} inferred from peaks and n_fold."
    )
    assert all_intensities.shape[2] == peaks.shape[0], (
        "Last dimension of all_intensities must match number of peaks."
    )

    # ----------------- build effective intensities ----------------- #
    I_eff = all_intensities.astype(float)

    # exponent on intensity
    if intensity_power != 1.0:
        I_eff = np.power(I_eff, intensity_power)

    # optional global clipping (percentile)
    if intensity_percentile is not None:
        vals = I_eff.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            vmax = np.percentile(vals, intensity_percentile)
            if vmax > 0:
                I_eff = np.clip(I_eff, 0.0, vmax)

    # ----------------- group weights per order ----------------- #
    # group_weight[o, y, x] = sum_{k in group o} I_eff[y, x, k]
    group_weights = np.zeros((num_orders, Ny, Nx), dtype=float)

    for o in range(num_orders):
        sl = slice(o * n_fold, (o + 1) * n_fold)
        group_weights[o] = np.sum(I_eff[:, :, sl], axis=2)

    # Denominator: sum over all groups (equiv. sum over all peaks)
    denominator = np.sum(group_weights, axis=0)  # (Ny, Nx)

    # ----------------- build numerator ----------------- #
    final_strain_maps = np.zeros((4, Ny, Nx), dtype=float)
    numerator = np.zeros_like(final_strain_maps)

    for o in range(num_orders):
        gw = group_weights[o]  # (Ny, Nx)
        for s in range(4):
            numerator[s] += strain_maps[o, s] * gw

    # ----------------- normalize, apply masking ----------------- #
    # global threshold for low-intensity pixels
    if min_total_intensity_frac > 0.0:
        T_max = np.nanmax(denominator) if np.isfinite(denominator).any() else 0.0
        thresh = min_total_intensity_frac * T_max
        valid_mask = denominator > max(thresh, eps)
    else:
        valid_mask = denominator > eps

    with np.errstate(divide='ignore', invalid='ignore'):
        final_strain_maps = np.divide(
            numerator,
            denominator[np.newaxis, :, :],
            out=np.zeros_like(numerator),
            where=valid_mask[np.newaxis, :, :]
        )

    return final_strain_maps

#%% The main 4D-STEM object

class HyperData:
    
    def __init__(self, data):
        
        # Read dataset from file path if input object is string
        if type(data) == str:
            data = read_4D(data)
        
        self.array = data
        self.ndim = data.ndim
        self.shape = data.shape
        self.real_shape = (data.shape[0], data.shape[1])
        self.k_shape = (data.shape[-2], data.shape[-1])
        self.dtype = data.dtype
        self.denoiser = Denoiser(self.array)


    def swap_domains(self):
        """
        Swap the real space and reciprocal space coordinates in the 4D dataset.
        For a dataset with dimensions (A, B, C, D),
        this method swaps them to (C, D, A, B).
        
        Returns:
        - swapped_data (numpy.ndarray): The 4D dataset with swapped dimensions.
        """
        
        # The original order is (0, 1, 2, 3) and we want to change to (2, 3, 0, 1)
        swapped_data = np.transpose(self.array, (2, 3, 0, 1))
        
        return HyperData(swapped_data)
    
    
    def alignment(self, r_center=5, iterations=1, returnStats=False):
        """
        Align the diffraction patterns through the Center of mass of the center beam
        
        Function written by Chuqiao Shi (2022)
        See on GitHub: Chuqiao2333/Hierarchical_Clustering
        
        Modified by Adan J Mireles (April, 2024)
        - Reordered dimensions (x, y) to (y, x)
        - Modified translation from 'ky//2 + com' to '(ky-1)//2 + com'
        """

        y, x, ky, kx = self.shape
        com_y, com_x = self._quickCOM(r_mask=r_center) 
        cbed_tran = np.copy(self.array)
        std_com = (np.std(com_y), np.std(com_x))
        mean_com = (np.mean(com_y), np.mean(com_x))
        
        print(f'Initial standard deviation statistics (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
        print(f'Initial COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')
        
        for idx in range(iterations):
            print()
            print(f'Processing {y} × {x} real-space positions. Iteration ({idx+1}/{iterations})...')        
            for i in tqdm(range(y), desc = 'Alignment Progress'):
                for j in range(x):
                    afine_tf = transform.AffineTransform(translation=(com_y[i, j] - (ky-1)/2, com_x[i, j] - (kx-1)/2))
                    cbed_tran[i,j,:,:] = transform.warp(cbed_tran[i,j,:,:], inverse_map=afine_tf)
        
            cbed_tran_Obj = HyperData(cbed_tran)
            com_y, com_x = cbed_tran_Obj._quickCOM()
            std_com = (np.std(com_y), np.std(com_x))
            mean_com = (np.mean(com_y), np.mean(com_x))
            
            print(f'Standard deviation statistics (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
            print(f'COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')
        
        if returnStats:
            return cbed_tran_Obj, mean_com, std_com
        else:
            return cbed_tran_Obj
    
    # TODO: add functionality for 3D array
    def rotate_dps(self, angle, units='deg'):
        """
        Rotate clockwise all diffraction patterns by angle in degrees
        """
        
        if units == 'rad':
            angle = np.degrees(angle)
        
        if self.ndim == 4:
            
            y, x, ky, kx = self.shape
            ky, kx = rotate(self.array[0, 0], angle).shape
            new_data = np.zeros((y, x, ky, kx), dtype=self.dtype)
            
            for y_idx in tqdm(range(y), desc = 'Rotating Diffraction Patterns'):
                for x_idx in range(x):
                    new_data[y_idx, x_idx] = rotate(self.array[y_idx, x_idx], angle)
                    
    
        return HyperData(new_data)
    
    def standardize(self, method='local'):
        """
        Standardize the dataset using NumPy.
    
        Supports both 4D data (A, B, C, D) and 3D data (N, A, B):
    
        - For 4D data, the last two axes (C, D) are treated as the diffraction
          pattern dimensions, and the first two (A, B) as the real-space grid.
        - For 3D data, the last two axes (A, B) are treated as the diffraction
          pattern dimensions, and the first axis (N) indexes patterns.
    
        Parameters
        ----------
        method : str, optional
            The standardization method to use. Options are:
            - 'global': Standardize the dataset globally
              (entire dataset has zero mean and unit variance).
            - 'local': Standardize each individual image in the dataset
              (each pattern on the last two axes has zero mean and unit variance).
            Default is 'global'.
    
        Returns
        -------
        HyperData
            The standardized dataset with zero mean and unit variance, either
            globally or locally.
    
        Raises
        ------
        ValueError
            If an invalid method is specified, or if local standardization is
            applied to data with fewer than 2 dimensions.
        """
        methods = ['global', 'local']
        if method not in methods:
            raise ValueError(f"Valid methods are: {methods}")
    
        arr = self.array
    
        if method == 'global':
            # Global standardization: calculate mean and std over the entire dataset
            mean = np.mean(arr)
            std = np.std(arr)
            standardized_tensor = (arr - mean) / (std + 1)
    
        elif method == 'local':
            # Local standardization: standardize each pattern defined by the last two axes
            if arr.ndim < 2:
                raise ValueError("Local standardization requires at least 2 dimensions")
    
            # Compute mean and std for each image on the last two dimensions
            axes = tuple(range(arr.ndim - 2, arr.ndim))  # last two axes
            mean = np.mean(arr, axis=axes, keepdims=True)
            std = np.std(arr, axis=axes, keepdims=True)
    
            # Standardize each image independently
            standardized_tensor = (arr - mean) / (std + 1)
    
        return HyperData(standardized_tensor)

    def normalize(self, method='global'):
        """
        Normalize the dataset to the range [0, 1] using NumPy.
    
        Supports both 4D data (A, B, C, D) and 3D data (N, A, B):
    
        - For 4D data, the last two axes (C, D) are treated as the diffraction
          pattern dimensions, and the first two (A, B) as the real-space grid.
        - For 3D data, the last two axes (A, B) are treated as the diffraction
          pattern dimensions, and the first axis (N) indexes patterns.
    
        Parameters
        ----------
        method : str, optional
            The normalization method to use. Options are:
            - 'global': Normalize using the global minimum and maximum of the
              entire dataset (one min/max for the whole array).
            - 'local': Normalize each individual pattern independently, using
              min/max over the last two axes (each pattern maps to [0, 1]).
            Default is 'global'.
    
        Returns
        -------
        HyperData
            New HyperData instance containing the normalized dataset with
            values in the range [0, 1] (up to numerical precision).
    
        Raises
        ------
        ValueError
            If an invalid method is specified, or if local normalization is
            applied to data with fewer than 2 dimensions.
        """
        methods = ['global', 'local']
        if method not in methods:
            raise ValueError(f"Valid methods are: {methods}")
    
        arr = self.array
    
        if method == 'global':
            # Global normalization: single min/max over entire dataset
            min_val = np.min(arr)
            max_val = np.max(arr)
            denom = max_val - min_val
    
            if denom == 0:
                # Constant array: return zeros
                normalized_tensor = np.zeros_like(arr, dtype=np.float32)
            else:
                normalized_tensor = (arr - min_val) / denom
    
        elif method == 'local':
            # Local normalization: per-pattern min/max on the last two axes
            if arr.ndim < 2:
                raise ValueError("Local normalization requires at least 2 dimensions")
    
            axes = tuple(range(arr.ndim - 2, arr.ndim))  # last two axes
    
            min_val = np.min(arr, axis=axes, keepdims=True)
            max_val = np.max(arr, axis=axes, keepdims=True)
            denom = max_val - min_val
    
            # Avoid division by zero: where denom == 0, set denom to 1 (pattern is constant)
            denom_safe = np.where(denom == 0, 1, denom)
    
            normalized_tensor = (arr - min_val) / denom_safe
    
        return HyperData(normalized_tensor)

    
    def clip(self, a_min=1, a_max=None):
        """
        Clip data values in the 4D dataset to the interval [a_min, a_max].
    
        Values smaller than `a_min` are set to `a_min`; values larger than `a_max`
        are set to `a_max`. If `a_min` or `a_max` is None, clipping on that side
        is skipped.
    
        Parameters
        ----------
        a_min : float or None, optional
            Lower bound for clipping. If None, no lower clipping is applied.
            Default is 1.
        a_max : float or None, optional
            Upper bound for clipping. If None, no upper clipping is applied.
            Default is None.
    
        Returns
        -------
        HyperData
            New HyperData instance with clipped data values.
        """
        return HyperData(clip_values(self.array, a_min, a_max))

    
    def crop(self, ylim=None, xlim=None, kylim=None, kxlim=None,
             kshape=None, rshape=None):
        """
        Crop a 4D dataset either in the real or the reciprocal domain,
        with optional subpixel cropping and enforced resizing in both
        k-space (kshape) and real space (rshape).
    
        Parameters
        ----------
        ylim : int or tuple, optional
            Real-space vertical limits.
        xlim : int or tuple, optional
            Real-space horizontal limits.
        kylim : int or tuple of (float or int), optional
            Reciprocal-space vertical limits. Can be float for subpixel cropping.
        kxlim : int or tuple of (float or int), optional
            Reciprocal-space horizontal limits. Can be float for subpixel cropping.
        kshape : tuple of (int, int), optional
            Shape (A, B) to resize cropped diffraction patterns to.
            If not provided, it is inferred from the cropped reciprocal region
            when subpixel cropping or explicit k-limits are used.
        rshape : tuple of (int, int), optional
            Shape (Ny_out, Nx_out) for real-space resizing (generalized binning).
            - If rshape divides the current real-space shape, block-averaging
              (binning) is used.
            - Otherwise, bilinear interpolation is used along the real-space axes.
        """
    
        def parse_limits(limits, max_length, allow_float=False, name="limits"):
            """Return (start, end) within [0, max_length]."""
            if limits is None:
                return (0, max_length)
    
            # Single index
            if isinstance(limits, (Integral, np.integer)):
                idx = int(limits)
                if idx < 0 or idx >= max_length:
                    raise ValueError(f"{name}: index {idx} out of bounds for axis length {max_length}")
                return (idx, idx + 1)
    
            # Two-element sequence
            if isinstance(limits, (tuple, list)) and len(limits) == 2:
                start, end = limits
    
                if not allow_float:
                    # Real-space indices must be integers
                    if isinstance(start, float) or isinstance(end, float):
                        raise ValueError(f"{name}: real-space limits must be integers")
                    start = int(start)
                    end = int(end)
    
                if start < 0 or end > max_length:
                    raise ValueError(
                        f"{name}: range ({start}, {end}) out of bounds for axis length {max_length}"
                    )
                if end <= start:
                    raise ValueError(f"{name}: end ({end}) must be greater than start ({start})")
                return (start, end)
    
            raise ValueError(f"{name}: limits must be int, tuple, list, or None")
    
        Ny, Nx, Ky, Kx = self.shape
    
        # --- Real-space limits ---
        ylim_range = parse_limits(ylim, Ny, allow_float=False, name="ylim")
        xlim_range = parse_limits(xlim, Nx, allow_float=False, name="xlim")
    
        # Fast path: only real-space crop, no k-space crop or resizing, no r-resize
        if kylim is None and kxlim is None and kshape is None and rshape is None:
            y0r, y1r = ylim_range
            x0r, x1r = xlim_range
            return HyperData(self.array[y0r:y1r, x0r:x1r])
    
        # --- Reciprocal-space limits (allow floats for subpixel cropping) ---
        kylim_range = parse_limits(kylim, Ky, allow_float=True, name="kylim")
        kxlim_range = parse_limits(kxlim, Kx, allow_float=True, name="kxlim")
    
        # Integer boundaries for actual array slicing (pad around subpixel ROI)
        y0 = int(np.floor(kylim_range[0]))
        y1 = int(np.ceil(kylim_range[1]))
        x0 = int(np.floor(kxlim_range[0]))
        x1 = int(np.ceil(kxlim_range[1]))
    
        if y0 < 0 or y1 > Ky or x0 < 0 or x1 > Kx:
            raise ValueError(
                f"Reciprocal-space crop out of bounds: "
                f"y in [{y0}, {y1}), x in [{x0}, {x1}), shape ({Ky}, {Kx})"
            )
    
        # Determine whether subpixel cropping is needed (any float in k-limits)
        def is_int_like(v):
            return isinstance(v, (Integral, np.integer))
    
        subpixel = not (
            is_int_like(kylim_range[0])
            and is_int_like(kylim_range[1])
            and is_int_like(kxlim_range[0])
            and is_int_like(kxlim_range[1])
        )
    
        # Natural reciprocal-space shape from the slice
        natural_kshape = (y1 - y0, x1 - x0)
    
        # Normalize kshape (if provided)
        if kshape is not None:
            if len(kshape) != 2:
                raise ValueError("kshape must be a tuple (A, B)")
            kshape = (int(kshape[0]), int(kshape[1]))
            if kshape[0] <= 0 or kshape[1] <= 0:
                raise ValueError("kshape must contain positive integers")
        else:
            # If user provided k-limits or we are doing subpixel cropping,
            # default kshape to the natural cropped size
            if kylim is not None or kxlim is not None or subpixel:
                kshape = natural_kshape
    
        if kshape is not None and kshape[0] != kshape[1]:
            print(f"Warning: Non-square diffraction pattern shape {kshape}. Proceeding anyway.")
    
        # --- Crop in real space first ---
        y0r, y1r = ylim_range
        x0r, x1r = xlim_range
        subarray = self.array[y0r:y1r, x0r:x1r]
    
        # --- Crop reciprocal region ---
        cropped = subarray[:, :, y0:y1, x0:x1]
    
        # --- k-space resize (subpixel or enforced kshape) ---
        needs_k_resize = subpixel or (kshape is not None and kshape != natural_kshape)
    
        if needs_k_resize:
            # If kshape is None here, fall back to natural_kshape
            if kshape is None:
                kshape = natural_kshape
    
            new_data = np.empty(
                (subarray.shape[0], subarray.shape[1], kshape[0], kshape[1]),
                dtype=subarray.dtype,
            )
    
            for i in range(subarray.shape[0]):
                for j in range(subarray.shape[1]):
                    dp = cropped[i, j]
                    new_data[i, j] = transform.resize(
                        dp,
                        kshape,
                        order=1,          # bilinear
                        mode="reflect",
                        anti_aliasing=True,
                    )
    
            cropped = new_data
    
        # --- Real-space resize (rshape) ---
        if rshape is not None:
            if len(rshape) != 2:
                raise ValueError("rshape must be a tuple (Ny_out, Nx_out)")
    
            Ny_out, Nx_out = int(rshape[0]), int(rshape[1])
            if Ny_out <= 0 or Nx_out <= 0:
                raise ValueError("rshape must contain positive integers")
    
            Ny_c, Nx_c, Ky_c, Kx_c = cropped.shape
    
            # If already at requested size, nothing to do
            if (Ny_out, Nx_out) != (Ny_c, Nx_c):
    
                # Case 1: exact divisors -> block-averaging / binning
                if Ny_c % Ny_out == 0 and Nx_c % Nx_out == 0:
                    by = Ny_c // Ny_out
                    bx = Nx_c // Nx_out
                    # Reshape and average over bin axes
                    reshaped = cropped.reshape(
                        Ny_out, by,
                        Nx_out, bx,
                        Ky_c, Kx_c
                    )
                    # Average over the binning dimensions (1 and 3)
                    binned = reshaped.mean(axis=(1, 3))
                    cropped = binned.astype(cropped.dtype, copy=False)
    
                else:
                    # Case 2: non-divisors -> interpolation along real-space axes
                    # Treat each diffraction pattern (Ky*Kx) as "channels"
                    channels = Ky_c * Kx_c
                    tmp = cropped.reshape(Ny_c, Nx_c, channels)
    
                    resized = transform.resize(
                        tmp,
                        (Ny_out, Nx_out, channels),
                        order=1,          # bilinear in real space
                        mode="reflect",
                        anti_aliasing=True,
                    )
                    cropped = resized.reshape(Ny_out, Nx_out, Ky_c, Kx_c).astype(
                        cropped.dtype, copy=False
                    )
    
        return HyperData(cropped)
           
    
    # Private method: helper function for the 'alignment' method
    def _quickCOM(self, r_mask=5):
        """
        Compute the center of mass (COM) of electron diffraction patterns within a 
        4D-STEM dataset.
        
        Function based on that written by Chuqiao Shi (2022)
        See on GitHub: Chuqiao2333/Hierarchical_Clustering
        
        Inputs:
            cbed_data: 4D numpy array
            r_mask   : radius of mask used for COM calculation (int or float) 
        Outputs:
            ap2_y, ap2_x : numpy arrays containing the x and y coordinates of the 
            centers of mass for each position in the real-space grid.
        """
        
        y, x, ky, kx = np.shape(self.array)
        center_x = (kx-1) // 2
        center_y = (ky-1) // 2

        # Assuming make_mask function is defined elsewhere that creates a circular mask
        if type(r_mask) == tuple:
            inner_mask = make_mask((center_y, center_x), r_mask[0], mask_dim=(ky, kx), invert=True)
            outer_mask = make_mask((center_y, center_x), r_mask[1], mask_dim=(ky, kx))
            mask = np.logical_and(inner_mask, outer_mask)
            
        else:
            mask = make_mask((center_y, center_x), r_mask, mask_dim=(ky, kx))
        
        ap2_x = np.zeros((y, x))
        ap2_y = np.zeros_like(ap2_x)
        vx = np.arange(kx)
        vy = np.arange(ky)

        for i in tqdm(range(y), desc = 'Computing centers of mass'):
            for j in range(x):
                cbed = np.squeeze(self.array[i, j, :, :] * mask)
                pnorm = np.sum(cbed)
                if pnorm != 0:
                    ap2_y[i, j] = np.sum(vy * np.sum(cbed, axis=0)) / pnorm
                    ap2_x[i, j] = np.sum(vx * np.sum(cbed, axis=1)) / pnorm

        return ap2_y, ap2_x
    
    def fix_elliptical_distortions(self, r=None, R=None, interp_method='linear', 
                                   show_ellipse=False, return_fix=True, **kwargs):
        """
        Corrects elliptical distortions across all diffraction patterns in the dataset.
    
        This function computes the average diffraction pattern, fits an ellipse to this average,
        and applies an affine transformation to each diffraction pattern (by transforming the 
        ellipse into a circle) to correct the elliptical distortion.
        
        Parameters
        ----------
        r : int, optional
            Inner radius of the annular mask. Defaults to one fifth of the diffraction pattern's horizontal dimension if not specified.
        R : int, optional
            Outer radius of the annular mask. Defaults to two fifths of the diffraction pattern's horizontal dimension if not specified.
        interp_method : str, optional
            Interpolation method to use during affine transformations. Can be 'linear' or 'cubic'.
        show_ellipse : bool, optional
            Plots the mean diffraction pattern in the dataset with an overlaid ring of inner radius r and outer radius R. If return_fix 
            is False, it plots the ring in the non-corrected pattern. Otherwise, it plots it in the corrected pattern.
        
        Returns
        -------
        HyperData
            A new instance of HyperData containing the corrected data.
    
        Notes
        -----
        This method must be applied after applying the 'alignment' method for better accuracy.
        """
        
        A, B, C, D = self.shape
        
        if not r:
            r = C//5 
        if not R:
            R = 2*C//5
        
        mean_pattern = self.get_dp('mean').array
        params = self._extract_ellipse(mean_pattern, C//2, D//2, r, R)
        corrected_data = np.empty_like(self.array)
        
        Ang, a, b = params 
        print("Ellptical statistics before correction:")
        print(f"Ellipse rotation = {np.degrees(Ang)} degrees \nMajor axis 'a' = {a} px \nMinor axis 'b' = {b} px \n")
        
        if show_ellipse:
            self._plot_with_ring(np.log(mean_pattern), r, R, **kwargs)
        
        # Perform Correction
        for i in tqdm(range(A), desc="Fixing elliptical distortions"):
            for j in range(B):
                corrected_data[i, j] = self._apply_affine_transformation(self.array[i, j], *params, interp_method)
        
        corrected_data = HyperData(corrected_data)
        
        mean_pattern = corrected_data.get_dp('mean').array
        params = corrected_data._extract_ellipse(mean_pattern, C//2, D//2, r, R)
        
        Ang, a, b = params 
        print("Ellptical statistics after correction:")
        print(f"Ellipse rotation = {np.degrees(Ang)} degrees \nMajor axis 'a' = {a} px \nMinor axis 'b' = {b} px \n")
        
        if show_ellipse:
            self._plot_with_ring(np.log(mean_pattern), r, R, **kwargs)
        
        return corrected_data

    #TODO: verify that the cost function is good
    def _extract_ellipse(self, image, cy, cx, r, R):
        """
        Fit an ellipse to an annular region of an image (diffraction pattern).
    
        The function calculates parameters for the ellipse that best fits the specified annular region
        of the mean diffraction pattern. These parameters are optimized to minimize the defined cost function.
    
        Parameters
        ----------
        image : numpy.ndarray
            The mean diffraction pattern as a 2D numpy array.
        cy : int
            The y-coordinate of the center of the diffraction pattern.
        cx : int
            The x-coordinate of the center of the diffraction pattern.
        r : int
            Inner radius for the annular mask.
        R : int
            Outer radius for the annular mask.
    
        Returns
        -------
        tuple
            A tuple of fitted ellipse parameters (A, a, b), where A is the rotation angle,
            a is the length of the major axis, and b is the length of the minor axis.
        """
        y, x = np.indices(image.shape)  # Get array indices for the entire image
        mask = ((x - cx)**2 + (y - cy)**2 >= r**2) & ((x - cx)**2 + (y - cy)**2 <= R**2)
        x_vals, y_vals = x[mask], y[mask]  # Get the coordinates within the mask
        intensities = image[mask]  # Get the intensities at these coordinates

        def ellipse_cost(params):
            A, a, b = params
            cost = intensities**4 * (((x_vals - cx) * np.cos(A) + (y_vals - cy) * np.sin(A))**2 / a**2 +
                                  ((y_vals - cy) * np.cos(A) - (x_vals - cx) * np.sin(A))**2 / b**2 - 1)**2
            return np.sum(cost)

        initial_guess = [0, (R-r)/2 + 1, (R-r)/2 - 1]  # Initial guess for A, a, b
        result = minimize(ellipse_cost, initial_guess, method='L-BFGS-B')
        return result.x


    def _apply_affine_transformation(self, image, A, a, b, interp_method):
        """
        Apply an affine transformation to make the fitted ellipse a circle.

        Parameters
        ----------
        image : numpy.ndarray
            The 2D diffraction pattern to correct.
        A : float
            The angle of rotation of the ellipse.
        a : float
            The length of the major axis.
        b : float
            The length of the minor axis.

        Returns
        -------
        numpy.ndarray
            The corrected diffraction pattern.
        """
        cols, rows = image.shape
        
        # Points in the original image
        p1 = np.float32([
            [a * np.cos(A) + cols//2, a * np.sin(A) + rows//2],
            [-a * np.cos(A) + cols//2, -a * np.sin(A) + rows//2],
            [b * np.sin(A) + cols//2, -b * np.cos(A) + rows//2]
        ])
        
        # Depending on whether a or b is larger, set the corresponding points in the corrected image
        # This ensures that the affine transformation results in expansion rather than shrinkage of the
        # transformed diffraction pattern.
        if a < b:
            p2 = np.float32([
                [b * np.cos(A) + cols//2, b * np.sin(A) + rows//2],
                [-b * np.cos(A) + cols//2, -b * np.sin(A) + rows//2],
                [b * np.sin(A) + cols//2, -b * np.cos(A) + rows//2]
            ])
        else:
            p2 = np.float32([
                [a * np.cos(A) + cols//2, a * np.sin(A) + rows//2],
                [-a * np.cos(A) + cols//2, -a * np.sin(A) + rows//2],
                [a * np.sin(A) + cols//2, -a * np.cos(A) + rows//2]
            ])
            
        M = cv2.getAffineTransform(p1, p2)
        
        if interp_method == 'cubic':
            transformed_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_CUBIC)
            
        if interp_method == 'linear':
            transformed_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

        return transformed_image
    
    
    def _plot_with_ring(self, data, inner_radius, outer_radius, **kwargs):
        """
        Plots a 2D array with an overlaid red ring directly on the same figure.
        
        Parameters
        ----------
        data : numpy.ndarray
            The 2D array to plot.
        inner_radius : float
            The inner radius of the ring.
        outer_radius : float
            The outer radius of the ring.
        """
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Generate a grid of points
        y, x = np.indices(data.shape)
        center_x, center_y = np.array(data.shape) // 2
        
        # Calculate the radius for each point in the grid
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create a mask for the ring
        ring_mask = (radius >= inner_radius) & (radius <= outer_radius)
        
        # Create a colored overlay where the ring is red
        ring_overlay = np.zeros(data.shape + (4,), dtype=np.float32)  # Adding a new dimension for RGBA
        ring_overlay[..., 0] = 1  # Red channel
        ring_overlay[..., 3] = ring_mask * 1.0  # Alpha channel only set where the ring is
        
        # Display the image
        ax.imshow(data, **kwargs)
        ax.imshow(ring_overlay, cmap='hot', alpha=0.5) 
        
        # Set plot details
        ax.set_title('Mean diffraciton with annular overlay')
        ax.axis('off')
        
        plt.show()
    
    def centerBeam_Stats(self, square_side=8):
        """
        Analyze diffraction patterns to find the mean and standard deviation of 
        the center of mass.
    
        Parameters:
        data (numpy.ndarray): 4D dataset of shape (A, B, C, D).
        square_side (int): Side length of the square used to calculate the 
        center of mass.
    
        Returns:
        tuple: Mean and standard deviation of the center of mass coordinates.
        """
        A, B, C, D = self.shape
        com_coordinates = []
    
        # Define the region for center of mass calculation
        half_side = square_side // 2
        center = C//2
        min_coord, max_coord = center - half_side, center + half_side
    
        # Calculate center of mass for each BxB image
        for i in range(A):
            for j in range(B):
                region = self.array[i, j, min_coord:max_coord, min_coord:max_coord]
                com = center_of_mass(region)
                com_coordinates.append((com[0] + min_coord, com[1] + min_coord))
    
        # Convert to numpy array for ease of calculation
        com_coordinates = np.array(com_coordinates)
    
        # Calculate mean and standard deviation
        mean_com = np.mean(com_coordinates, axis=0)
        std_dev_com = np.std(com_coordinates, axis=0)
    
        print(f"Mean CoM Coordinate (ky, kx): {mean_com}")
        print(f"Standard Deviation of CoM Coordinates (ky, kx): {std_dev_com} px")
    
        return mean_com, std_dev_com
    
    #TODO: Is there a faster way to get this?
    def get_ewpc(self):
        """
        Transform dataset to Exit Wave Power Cepstrum (EWPC)
        """
    
        return HyperData(np.abs(fftshift(
                                         fft2(
                                              np.log(self.clip().array),
                                              axes=(-2, -1)
                                              ),
                                         axes=(-2, -1)
                                        )))
    
    
    def get_stdDev(self, domain='reciprocal'):
        """
        Calculate a 2D mask of the standard deviation of each pixel across the
        specified domain (real or reciprocal).

        Parameters:
        domain (string): the domain for which the standard deviation will be
        computer for each position.

        Returns:
        numpy array: 2D mask of standard deviations with same shape as specified domain.
        """
        # Validate the shape of the data
        if len(self.array.shape) != 4:
            raise ValueError("Data must be a 4D array")
                
        if domain == 'reciprocal':
            # Calculate the standard deviation for each pixel across all diffraction patterns
            std_dev = np.std(self.array, axis=(0, 1))
        elif domain == 'real':
            # Calculate the standard deviation for each pixel across all scanning positions
            std_dev = np.std(self.array, axis=(2, 3))
        else:
            raise ValueError("'domain' must be 'reciprocal' or 'real'")
            
        return ReciprocalSpace(std_dev)
    
    def get_dp(self, y=None, x=None, mask=None, operation=None, **flat_kwargs):
        """
        Obtain a diffraction pattern from a 3D/4D dataset, either at a
        specific point, averaged over a specified region, or via a special operation.
    
        Parameters
        ----------
        y : int | tuple[int, int] | None
            If tuple (ymin, ymax), averages over rows [ymin:ymax];
            if int, a single row; if None, ignored.
        x : int | tuple[int, int] | None
            If tuple (xmin, xmax), averages over cols [xmin:xmax];
            if int, a single column; if None, ignored.
        mask : ndarray[bool] | None
            Shape (A, B). If provided and operation in
            {'mean','median','max','min','std'}, returns aggregation over all
            True pixels. If operation == 'random', picks a random True pixel
            and returns its DP.
        reciprocal_mask : ndarray[bool] | None
            Shape (C, D). Currently only supports mean over masked k-space (legacy).
        operation : {'mean','median','max','min','std','random','flat_mean'} | None
            Operation to apply. Defaults to 'mean' if None.
            - 'mean','median','max','min','std' : aggregate over the selected
              real-space region (or globally if no region/mask is given).
            - 'random' : return a single randomly selected diffraction pattern.
            - 'flat_mean' : special two-step operation based on radial masking
              and a flat-field style threshold (see implementation).
    
        Returns
        -------
        ReciprocalSpace or ndarray
        """
    
        # Helper to parse ranges given int/tuple/None
        def _parse_range(val, max_len, name):
            if isinstance(val, tuple):
                a, b = val
                if not (0 <= a < b <= max_len):
                    raise ValueError(f"Invalid {name} range ({a}, {b}) for length {max_len}.")
                return a, b
            elif isinstance(val, int):
                if not (0 <= val < max_len):
                    raise ValueError(f"{name} index {val} out of bounds for length {max_len}.")
                return val, val + 1
            elif val is None:
                return 0, max_len
            else:
                raise ValueError(f"{name} must be int, tuple, or None.")
    
        operations = {
            'mean':   np.mean,
            'median': np.median,
            'max':    np.max,
            'min':    np.min,
            'std':    np.std,
            'random': 'random',
            'flat_mean': None,
        }
    
        # Default operation is mean
        if operation is None:
            operation = 'mean'
    
        if operation not in operations:
            valid_ops = ', '.join(f"'{op}'" for op in operations.keys())
            raise ValueError(f"'operation' must be one of: {valid_ops}.")
    
        # ----------------------- RANDOM SELECTION ------------------------ #
        if operation == 'random':
            rng = np.random
            if self.ndim not in (3, 4):
                raise ValueError("Random selection supported only for 3D or 4D arrays.")
    
            # (2) mask constraint
            if mask is not None:
                if self.ndim != 4:
                    raise ValueError("mask requires a 4D dataset (A,B,C,D).")
                if mask.shape != (self.shape[0], self.shape[1]):
                    raise ValueError(f"'mask' must be shape {(self.shape[0], self.shape[1])}.")
                ys, xs = np.where(mask)
                if ys.size == 0:
                    raise ValueError("mask has no True pixels.")
                idx = rng.randint(0, ys.size)
                y_pos, x_pos = int(ys[idx]), int(xs[idx])
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos}) using mask...\n")
                return ReciprocalSpace(self.array[y_pos, x_pos])
    
            # (1) region via y/x, at least one must be a tuple (4D only)
            if self.ndim == 4 and (isinstance(y, tuple) or isinstance(x, tuple)):
                A, B = self.shape[0], self.shape[1]
                y0, y1 = _parse_range(y, A, 'y')
                x0, x1 = _parse_range(x, B, 'x')
    
                if not (isinstance(y, tuple) or isinstance(x, tuple)):
                    raise ValueError("When constraining random by region, at least one of y or x must be a tuple.")
    
                if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                    raise ValueError("Empty region for random selection.")
    
                y_pos = rng.randint(y0, y1)
                x_pos = rng.randint(x0, x1)
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos}) within specified region...\n")
                return ReciprocalSpace(self.array[y_pos, x_pos])
    
            # Global random selection
            if self.ndim == 4:
                y_pos = int(self.shape[0] * rng.random())
                x_pos = int(self.shape[1] * rng.random())
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos})...\n")
                return ReciprocalSpace(self.array[y_pos, x_pos])
            else:  # self.ndim == 3
                idx = int(self.shape[0] * rng.random())
                print(f"Acquired diffraction pattern of index {idx}...\n")
                return ReciprocalSpace(self.array[idx])
    
        # ------------------------ FLAT MEAN ------------------------------ #
        if operation == 'flat_mean':
            r = (self.shape[-2] + self.shape[-1]) / 4
            r_min = 0.6 * r
            r_max = 0.8 * r
            reduced_data = self.apply_mask(r_min, r_max,)
            mean, flat_mask = mask_and_average(
                reduced_data.array,
                return_mask=True,
                function='sum_2d',
                threshold='upper',
                **flat_kwargs
            )
            result = np.mean(self.apply_mask(mask=flat_mask, domain='real',).array, axis=0)
            return ReciprocalSpace(result)
    
        # From here on: mean/median/max/min
        agg_func = operations[operation]
    
        # ------------------------ REAL MASK ------------------------------ #
        if mask is not None:
            if self.ndim != 4:
                raise ValueError("mask requires a 4D dataset (A,B,C,D).")
            if mask.shape != (self.shape[0], self.shape[1]):
                raise ValueError(f"'mask' must be shape {(self.shape[0], self.shape[1])}.")
            sub = self.array[mask]   # shape (N, C, D)
            result = agg_func(sub, axis=0)
            return ReciprocalSpace(result)
    
        # -------------------- REGION VIA (y, x) -------------------------- #
        if y is not None or x is not None:
            if self.ndim == 4:
                A, B = self.shape[0], self.shape[1]
    
                # Convenience: map to same forms as original code
                if isinstance(y, tuple) and isinstance(x, tuple):
                    ymin, ymax = y
                    xmin, xmax = x
                    sub = self.array[ymin:ymax, xmin:xmax, :, :]
                    result = agg_func(sub, axis=(0, 1))
                    return ReciprocalSpace(result)
    
                elif isinstance(y, tuple) and isinstance(x, int):
                    ymin, ymax = y
                    sub = self.array[ymin:ymax, x, :, :]
                    result = agg_func(sub, axis=0)
                    return ReciprocalSpace(result)
    
                elif isinstance(y, int) and isinstance(x, tuple):
                    xmin, xmax = x
                    sub = self.array[y, xmin:xmax, :, :]
                    result = agg_func(sub, axis=0)
                    return ReciprocalSpace(result)
    
                elif isinstance(y, int) and isinstance(x, int):
                    # Single DP; operation degenerates to identity
                    return ReciprocalSpace(self.array[y, x])
    
                elif isinstance(y, tuple) and x is None:
                    ymin, ymax = y
                    sub = self.array[ymin:ymax, :, :, :]
                    result = agg_func(sub, axis=(0, 1))
                    return ReciprocalSpace(result)
    
                elif isinstance(y, int) and x is None:
                    # Aggregate along x with chosen operation
                    sub = self.array[y, :, :, :]
                    result = agg_func(sub, axis=0)
                    return ReciprocalSpace(result)
    
                elif y is None and isinstance(x, tuple):
                    xmin, xmax = x
                    sub = self.array[:, xmin:xmax, :, :]
                    result = agg_func(sub, axis=(0, 1))
                    return ReciprocalSpace(result)
    
                elif y is None and isinstance(x, int):
                    sub = self.array[:, x, :, :]
                    result = agg_func(sub, axis=0)
                    return ReciprocalSpace(result)
    
                else:
                    raise ValueError(
                        "y and x must be either both tuples, both integers, "
                        "one tuple-one integer, or one integer-one None."
                    )
    
            elif self.ndim == 3:
                # For 3D: only support picking by index, as before
                if isinstance(y, int) and x is None:
                    return ReciprocalSpace(self.array[y, :, :])
                if y is None and isinstance(x, int):
                    return ReciprocalSpace(self.array[x, :, :])
                raise ValueError("Region-based (tuple) selection currently only supported for 4D arrays.")
    
        # ---------------------- GLOBAL OPERATIONS ------------------------ #
        # If no y/x or masks are given, apply global aggregation over (A,B)
        if self.ndim == 4:
            result = agg_func(self.array, axis=(0, 1))
            return ReciprocalSpace(result)
        elif self.ndim == 3:
            result = agg_func(self.array, axis=0)
            return ReciprocalSpace(result)
    
        raise ValueError("No parameters specified.")


    #TODO: Generalize this to reshape the real-space to whatever size is desired
    #      if the desired shape is a divisor of the original dimensions, bin n-times.
    def bin_data(self, bin_domain='real', iterations=1):
        """
        Bin 4D dataset by averaging over either the first two dimensions (real space) or the last two dimensions (reciprocal space).
        """
        
        assert len(self.shape) == 4, "'dataset' must be four-dimensional"
        assert isinstance(iterations, int) and iterations >= 1, "'iterations' must be an integer >= 1"
        
        bin_factor = 2 ** iterations
        
        if bin_domain == 'real':
            A, B, C, D = self.shape[0] // bin_factor, self.shape[1] // bin_factor, self.shape[2], self.shape[3]
            binned_dataset = np.zeros((A, B, C, D))
            
            for i in range(A):
                for j in range(B):
                    # Average over the first two dimensions (real space)
                    binned_dataset[i, j] = np.mean(self.array[bin_factor*i:bin_factor*(i+1), bin_factor*j:bin_factor*(j+1)], axis=(0, 1))
    
        elif bin_domain == 'reciprocal':
            A, B, C, D = self.shape[0], self.shape[1], self.shape[2] // bin_factor, self.shape[3] // bin_factor
            binned_dataset = np.zeros((A, B, C, D))
            
            for i in range(C):
                for j in range(D):
                    # Average over the last two dimensions (reciprocal space)
                    binned_dataset[:, :, i, j] = np.mean(self.array[:, :, bin_factor*i:bin_factor*(i+1), bin_factor*j:bin_factor*(j+1)], axis=(2, 3))
    
        else:
            raise ValueError("Invalid bin_domain. Choose 'real' or 'reciprocal'.")
    
        return HyperData(binned_dataset)
    
    #TODO: make this return a RealSpace object
    #TODO: add the option to accept an arbitrary mask
    def get_virtualImage(self, r_minor, r_major, vmin=None, vmax=None, 
                         grid=True, num_div=10, plotMask=False, gridColor='black',
                         axes=True, returnDetector = False):
        """
        Plot virtual detector image
        - r_minor, r_major specify the dimensions of the annular ring over which to integrate
        - vmin, vmax specify the limiting pixel values to be plotted
        
        returnArrays: if True, returns the virtual image and detector mask
        """
        
        assert self.ndim == 4, "Input dataset must be 4-dimensional"
        
        A,B,C,D = self.shape
        
        # Virtual detector mask        
        detector_mask = make_mask(((C-1)/2, (D-1)/2), (r_minor, r_major), mask_dim=(C, D))
        
        masked_image = np.sum(self.apply_mask(r_minor, r_major, a_min=0).array, axis=(2,3))
            
        if vmin:
            vmin = vmin
        else:
            vmin = np.min(masked_image[masked_image>0])
        if vmax:
            vmax = vmax
        else:
            vmax = np.max(masked_image)
            
        plt.imshow(masked_image, vmin=vmin, vmax=vmax, cmap='gray')
        
        if type(num_div) == tuple:
            ydiv, xdiv = num_div
        else:
            ydiv = num_div
            xdiv = num_div
        
        if axes:
            plt.axis('on')
            plt.xticks(np.arange(0, B, B//xdiv))
            plt.yticks(np.arange(0, A, A//ydiv))
        else:
            plt.axis('off')    
    
        if grid:
            plt.grid(c=gridColor)
        plt.show()
        
        if plotMask:
            plt.imshow(np.log(np.mean(self.array, axis=(0,1))*detector_mask), cmap='turbo',)
            plt.show()
    
        if returnDetector:
            return masked_image, np.mean(self.array, axis=(0,1))*detector_mask
        else:
            return masked_image
        
        
    def get_centers(self, r, ref_coords, method='CoM', real_mask=None):
        """
        Compute Bragg spot centers for each diffraction pattern.
    
        Supports both 4D (Ny×Nx×C×D) and 3D (B×C×D) HyperData.
    
        Parameters
        ----------
        r : float
            Radius (in pixels) of the local window used by the center-finding
            algorithm in each diffaction pattern.
        ref_coords : array-like or nested list
            Reference peak coordinates:
              - 4D, array input:
                  ndarray of shape (n_peaks, 2) used for every DP (same peaks).
              - 4D, ragged list-of-lists:
                  `ref_coords[i][j]` is array-like of shape (n_ij, 2) for each
                  real-space position (i, j).
              - 3D, array input:
                  ndarray of shape (B_peaks, 2) used for each DP index.
              - 3D, ragged list:
                  `ref_coords[i]` is array-like of shape (n_i, 2) for each DP.
    
        method : {'CoM', ...}, optional
            Center-finding method to be passed to the underlying 2D
            ReciprocalSpace.get_centers (e.g., 'CoM' for center-of-mass).
        real_mask : ndarray[bool] or None, optional
            Only used for 4D datasets. Boolean mask of shape (Ny, Nx) defining
            which real-space positions (i, j) should have centers computed.
            - If real_mask[i, j] is True:
                The center-finding algorithm is applied at (i, j).
            - If real_mask[i, j] is False:
                Centers are skipped and filled with zeros:
                  * 4D + array ref_coords:
                      all_centers[i, j] remains zeros (n_peaks, 2).
                  * 4D + ragged list-of-lists:
                      an array of zeros with shape like ref_coords[i][j] is stored.
            This accelerates center-finding by skipping unmasked positions.
            For 3D datasets, passing real_mask is not supported and raises an error.
    
        Returns
        -------
        centers :
            - 4D & array input -> ndarray (Ny, Nx, n_peaks, 2)
            - 4D & list input  -> list of lists of arrays
            - 3D & array input -> ndarray (B, n_peaks, 2)
            - 3D & list input  -> list of arrays
        """
        assert 2 < self.ndim < 5, "HyperData must be 3D or 4D"
    
        # 4D case
        if self.ndim == 4:
            Ny, Nx, _, _ = self.shape
    
            if real_mask is not None:
                real_mask = np.asarray(real_mask)
                if real_mask.shape != (Ny, Nx):
                    raise ValueError(
                        f"real_mask must have shape (Ny, Nx) = {(Ny, Nx)}, "
                        f"got {real_mask.shape}."
                    )
    
            # ragged list-of-lists input
            if isinstance(ref_coords, list):
                centers = []
                for i in tqdm(range(Ny), desc="Computing centers (4D)"):
                    row = []
                    for j in range(Nx):
                        coords_ij = np.asarray(ref_coords[i][j])
    
                        # If masked out, skip computation and fill zeros
                        if real_mask is not None and not real_mask[i, j]:
                            if coords_ij.size == 0:
                                c = coords_ij.reshape(0, 2)
                            else:
                                c = np.zeros_like(coords_ij, dtype=float)
                            row.append(c)
                            continue
    
                        dp = self.get_dp(i, j)
                        c = dp.get_centers(r=r,
                                           ref_coords=coords_ij,
                                           show=False,
                                           method=method)
                        row.append(c)
                    centers.append(row)
                return centers
    
            # fixed array input
            coords = np.asarray(ref_coords)
            n_peaks = coords.shape[0]
            all_centers = np.zeros((Ny, Nx, n_peaks, 2), dtype=float)
    
            for i in tqdm(range(Ny), desc="Computing centers (4D)"):
                for j in range(Nx):
                    # If masked out, leave zeros and skip
                    if real_mask is not None and not real_mask[i, j]:
                        continue
                    dp = self.get_dp(i, j)
                    all_centers[i, j] = dp.get_centers(r=r,
                                                       ref_coords=coords,
                                                       show=False,
                                                       method=method)
            return all_centers
    
        # 3D case
        else:
            if real_mask is not None:
                raise ValueError("real_mask is only supported for 4D datasets.")
    
            B, _, _ = self.shape
    
            # ragged list input
            if isinstance(ref_coords, list):
                centers = []
                for i in tqdm(range(B), desc="Computing centers (3D)"):
                    coords_i = np.asarray(ref_coords[i])
                    dp = self.get_dp(i)
                    c = dp.get_centers(r=r,
                                       ref_coords=coords_i,
                                       show=False,
                                       method=method)
                    centers.append(c)
                return centers
    
            # fixed array input
            coords = np.asarray(ref_coords)
            n_peaks = coords.shape[0]
            all_centers = np.zeros((B, n_peaks, 2), dtype=float)
            for i in tqdm(range(B), desc="Computing centers (3D)"):
                dp = self.get_dp(i)
                all_centers[i] = dp.get_centers(r=r,
                                                ref_coords=coords,
                                                show=False,
                                                method=method)
            return all_centers



    def get_intensities(self,
                        r=6,
                        centers=None,
                        ref_coords=None,
                        method='CoM',
                        compute_resBg=False,
                        residual_frac=0.9,
                        real_mask=None,
                        **resBg_kwargs):
        """
        Extract Bragg peak intensities from each diffraction pattern slice.
    
        Supports both 4D (Ny×Nx×C×D) and 3D (B×C×D) HyperData.
    
        Parameters
        ----------
        r : float, optional
            Integration radius in pixels.
        centers : array-like or list, optional
            If array:
              - shape (Ny, Nx, n_peaks, 2) for 4D
              - shape (B,  n_peaks, 2)     for 3D
            If list:
              - 4D: list of length Ny, each an inner list of length Nx of (n_ij, 2)
                arrays, so that `centers[i][j]` has shape (n_ij, 2) for DP (i, j).
              - 3D: list of length B of (n_i, 2) arrays, so that `centers[i]` has
                shape (n_i, 2) for DP i.
            If None, centers are computed via `self.get_centers(...)`, and
            `ref_coords` and `method` are forwarded there.
        ref_coords : array-like or nested list, optional
            Reference peak coordinates used when `centers is None`, passed to
            `self.get_centers(r, ref_coords=..., method=..., real_mask=...)`.
            See `get_centers` for allowed formats.
        method : {'CoM', ...}, optional
            Center-finding method to be passed to `get_centers` if `centers` is None.
        compute_resBg : bool, optional
            If True, estimate and subtract a residual background before integrating
            intensities in each diffraction pattern (forwarded to dp.get_intensities).
        residual_frac : float, optional
            Fraction of low-valued pixels used to estimate residual background
            (forwarded to dp.get_intensities).
        real_mask : ndarray[bool] or None, optional
            Only used for 4D datasets. Boolean mask of shape (Ny, Nx) defining
            which real-space positions (i, j) should have intensities computed.
            - If real_mask[i, j] is True:
                Intensities are computed at (i, j) using the provided centers.
            - If real_mask[i, j] is False:
                Intensities are skipped and filled with zeros:
                  * 4D + array centers:
                      all_ints[i, j, :] remains zeros (n_peaks,).
                  * 4D + ragged list-of-lists:
                      a 1D zero array of length equal to the number of centers
                      for that DP is stored.
            This can accelerate processing when only a subset of DPs are of interest.
            For 3D datasets, `real_mask` is not supported and will raise an error.
        **resBg_kwargs :
            Additional keyword arguments forwarded to `dp.get_intensities(...)`.
    
        Returns
        -------
        all_ints : np.ndarray or list
            If `centers` was an array (and all DPs share n_peaks), returns a
            fixed-shape ndarray:
              - (Ny, Nx, n_peaks) for 4D
              - (B,     n_peaks)  for 3D
    
            If `centers` was a list (ragged), returns a list of the same shape,
            where each entry is the 1D intensity array for that DP.
        """
        assert 2 < self.ndim < 5, "HyperData must be 3D or 4D"
    
        # compute centers if not provided
        if centers is None:
            centers = self.get_centers(
                r,
                ref_coords=ref_coords,
                method=method,
                real_mask=real_mask,
            )
    
        # helper to process a single DP
        def compute_dp_int(dp, dp_centers):
            return dp.get_intensities(
                r             = r,
                centers       = dp_centers,
                compute_resBg = compute_resBg,
                residual_frac = residual_frac,
                **resBg_kwargs
            )
    
        # --------------------------- 4D case ---------------------------- #
        if self.ndim == 4:
            Ny, Nx, _, _ = self.shape
    
            if real_mask is not None:
                real_mask = np.asarray(real_mask)
                if real_mask.shape != (Ny, Nx):
                    raise ValueError(
                        f"real_mask must have shape (Ny, Nx) = {(Ny, Nx)}, "
                        f"got {real_mask.shape}."
                    )
    
            # ragged (list-of-lists) branch
            if isinstance(centers, list):
                all_ints = []
                for i in tqdm(range(Ny), desc="Row"):
                    row_ints = []
                    for j in range(Nx):
                        dp_centers = np.asarray(centers[i][j])  # shape (n_ij, 2)
    
                        # If masked out, skip computation and fill zeros
                        if real_mask is not None and not real_mask[i, j]:
                            if dp_centers.size == 0:
                                row_ints.append(np.zeros(0, dtype=float))
                            else:
                                row_ints.append(np.zeros(dp_centers.shape[0], dtype=float))
                            continue
    
                        dp = self.get_dp(i, j)
                        row_ints.append(compute_dp_int(dp, dp_centers))
                    all_ints.append(row_ints)
                return all_ints
    
            # fixed-shape array branch
            else:
                centers_arr = np.asarray(centers)
                n_peaks = centers_arr.shape[-2]
                all_ints = np.zeros((Ny, Nx, n_peaks), dtype=float)
                for i in tqdm(range(Ny), desc="Calculating intensities"):
                    for j in range(Nx):
                        # If masked out, leave zeros and skip
                        if real_mask is not None and not real_mask[i, j]:
                            continue
                        dp_centers = centers_arr[i, j]  # (n_peaks, 2)
                        dp = self.get_dp(i, j)
                        all_ints[i, j, :] = compute_dp_int(dp, dp_centers)
                return all_ints
    
        # --------------------------- 3D case ---------------------------- #
        else:
            if real_mask is not None:
                raise ValueError("real_mask is only supported for 4D datasets.")
    
            B, _, _ = self.shape
    
            # ragged (list) branch
            if isinstance(centers, list):
                all_ints = []
                for i in tqdm(range(B), desc="DP"):
                    dp_centers = np.asarray(centers[i])  # shape (n_i, 2)
                    dp = self.get_dp(i)
                    all_ints.append(compute_dp_int(dp, dp_centers))
                return all_ints
    
            # fixed-shape array branch
            else:
                centers_arr = np.asarray(centers)
                n_peaks = centers_arr.shape[-2]
                all_ints = np.zeros((B, n_peaks), dtype=float)
                for i in tqdm(range(B), desc="Calculating intensities"):
                    dp_centers = centers_arr[i]  # (n_peaks, 2)
                    dp = self.get_dp(i)
                    all_ints[i, :] = compute_dp_int(dp, dp_centers)
                return all_ints

    
    def get_residualBg(self, 
                        r=6,
                        centers=None,  
                        ref_coords=None, 
                        method='CoM',
                        **resBg_kwargs
                        ):

        """
        Extract Bragg peak intensities from a single DP
        """
        
        assert 2 < self.ndim < 5, "Input dataset must be 3- or 4-dimensional"
        Ny, Nx, _, _ = self.shape
        
        if centers is None:
            centers = self.get_centers(r, ref_coords=ref_coords, method=method)
    
        residual_Bgs = np.zeros((Ny, Nx))
        
        for i in tqdm(range(Ny), desc="Computing residual backgrounds"):
            for j in range(Nx):
                
                dp = self.get_dp(i, j)
                residual_Bgs[i, j] = dp.get_residualBg(centers=centers[i,j], **resBg_kwargs)

                # if compute_resBg:
                #     res_bg = dp.get_residualBg(centers=centers[i, j], **resBg_kwargs)
                #     # if isinstance(r, (list, np.ndarray)):
                #     all_ints[i,j] -= res_bg * (np.pi*r**2) * residual_frac
                        
        return residual_Bgs
    
    #TODO: add docstring
    #TODO: this function seems to work for hexagonal lattices only. I need to extend
    #      to any number of peaks, maybe depends on symmetry. Add at least 6- and 4-fold symmetry
    #TODO: return as RealSpace object and add a `.show`  
    def get_strains(self, centers=None, ref_centers=None, ang=0, g_vector=1,
                    r_CoM=None, r_inner=None, r_outer=None, intensity_array=None, ewpc=False):
        """
        Calculate strain matrices using the given centers and reference centers.
        """
                
        def calculate_g_vectors(centers, g_vector):
            
            if centers.shape[2] == 6:
                g1 = centers[:, :, g_vector-1] - centers[:, :, g_vector+2]
                g2 = (centers[:, :, g_vector] + centers[:, :, g_vector+1]) / 2 - \
                     (centers[:, :, (g_vector+3) % 6] + centers[:, :, (g_vector+4) % 6]) / 2
                     
            return g1, g2

        def calculate_reference_g_vectors(ref_centers, g_vector):
            
            if ref_centers.shape[0] == 6:

                ref_g1 = ref_centers[g_vector-1] - ref_centers[g_vector+2]
                ref_g2 = (ref_centers[g_vector] + ref_centers[g_vector+1]) / 2 - \
                         (ref_centers[(g_vector+3) % 6] + ref_centers[(g_vector+4) % 6]) / 2
                         
            return ref_g1, ref_g2

        def calculate_strain_matrices(centers, ref_centers, ang_rad, g_vector, ewpc=ewpc):
            
            ydim, xdim = centers.shape[:2]
            exx = np.zeros((ydim, xdim))
            eyy = np.zeros((ydim, xdim))
            exy = np.zeros((ydim, xdim))
            erot = np.zeros((ydim, xdim))

            g1, g2 = calculate_g_vectors(centers, g_vector)
            ref_g1, ref_g2 = calculate_reference_g_vectors(ref_centers, g_vector)

            G_ref = np.array([[ref_g2[0], ref_g1[0]],
                              [ref_g2[1], ref_g1[1]]])

            R1 = np.array([[np.cos(ang_rad), np.sin(ang_rad)],
                           [-np.sin(ang_rad), np.cos(ang_rad)]])

            for i in tqdm(range(ydim), desc='Computing strain matrices'):
                for j in range(xdim):
                    G1 = np.array([[g2[i, j, 0], g1[i, j, 0]],
                                   [g2[i, j, 1], g1[i, j, 1]]])
                    
                    if not ewpc:
                        T = R1 @ G1 @ np.linalg.inv(G_ref) @ np.linalg.inv(R1)
                    else:
                        T = R1 @ G_ref @ np.linalg.inv(G1) @ np.linalg.inv(R1)
                        
                    R, U = polar(T)

                    eyy[i, j]  = 1 - U[0, 0]
                    exx[i, j]  = 1 - U[1, 1]
                    exy[i, j]  = U[1, 0]
                    erot[i, j] = np.arccos(R[0, 1])

            return exx, eyy, exy, erot

        def calculate_average_strain_maps(centers, ref_centers, ang_rad):
            
            ydim, xdim = centers.shape[:2]
            exx_maps = np.zeros((3, ydim, xdim))
            eyy_maps = np.zeros((3, ydim, xdim))
            exy_maps = np.zeros((3, ydim, xdim))
            erot_maps = np.zeros((3, ydim, xdim))

            for i, g_vector in enumerate(range(1, 4)):
                exx_maps[i], eyy_maps[i], exy_maps[i], erot_maps[i] = \
                    calculate_strain_matrices(centers, ref_centers, ang_rad, g_vector)

            exx = np.mean(exx_maps, axis=0)
            eyy = np.mean(eyy_maps, axis=0)
            exy = np.mean(exy_maps, axis=0)
            erot = np.mean(erot_maps, axis=0)

            return exx, eyy, exy, erot

        if ref_centers is None and centers is None:
            raise ValueError("Either 'ref_centers' or 'centers' must be defined.")

        if ref_centers is None:
            if r_inner is None or r_outer is None:
                r = (self.shape[-2] + self.shape[-1]) / 4
                reduced_data = self.apply_mask(r_inner=0.6 * r, r_outer=0.8 * r)
            else:
                reduced_data = self.apply_mask(r_inner, r_outer)
            mean, flat_mask = mask_and_average(reduced_data.array, return_mask=True, 
                                               function='sum_2d', threshold='upper', 
                                               percentile=5, show_mask=False)
            ref_centers = np.mean(centers[flat_mask], axis=0)

        if centers is None:
            centers = self.get_centers(r=r_CoM, ref_coords=ref_centers, method='CoM')

        ang_rad = np.radians(ang)

        if g_vector in [1, 2, 3]:
            return calculate_strain_matrices(centers, ref_centers, ang_rad, g_vector)
        elif g_vector is None:
            return calculate_average_strain_maps(centers, ref_centers, ang_rad)
        else:
            raise ValueError("Input 'g_vector' can only be one of the following: [1, 2, 3, None]")
        
        
    #TODO: add docstring and figure out what it is doing
    def apply_mask(self, r_inner=None, r_outer=None, a_min=0, mask=None, domain=None):
        
        if self.ndim < 3:
            raise ValueError("The data object must be of 3 or greater dimensions.")
        
        ky, kx = self.shape[-2], self.shape[-1]

        # Apply arbitrary mask on real or reciprocal space
        if mask is not None:
            if domain == 'real':
                self.array[~mask,:,:] = a_min
                return HyperData(self.array[~mask,:,:])
            elif domain == 'reciprocal':
                return HyperData(self.array * mask).clip(a_min=a_min)
        
        # For ring=shaped mask on reciprocal space
        elif r_outer is not None:
            bool_mask = make_mask(((ky-1)/2, (kx-1)/2), (r_inner, r_outer), mask_dim=(ky, kx),)
        
        # For center-beam removal
        else:            
            bool_mask = make_mask(((ky-1)/2, (kx-1)/2), r_inner, mask_dim=(ky, kx),)
        
        return HyperData(self.array * bool_mask).clip(a_min=a_min)
    
    #TODO: the output of this should be a real-space object
    def get_clusters(self, n_PCAcomponents, n_clusters, r_centerBeam, std_Threshold=0.2, power=1,
                     clustering_method="k-means", plotStdMask=True, plotScree=True, plotClusterMap=True,
                     plot3dClusterMap=False, filter_size=3, cluster_cmap=None, filter_iterations=1,
                     outer_ring=None, polar=False, split_disconnected=False):
        """
        Perform PCA-based dimensionality reduction and clustering on a 4D dataset,
        returning a cluster map with various clustering and visualization options.
    
        Parameters
        ----------
        n_PCAcomponents : int
            Number of principal components to retain for dimensionality reduction.
        n_clusters : int
            Number of clusters for clustering (initial cluster count).
        r_centerBeam : int
            Radius of the central region (beam) to mask out in the dataset.
        std_Threshold : float, optional
            Threshold for standard deviation filtering as a fraction of the maximum
            standard deviation. Pixels below this threshold are masked out.
        power : int, optional
            Exponent to apply to the dataset for emphasis
            (e.g., power=2 squares the values).
        clustering_method : {"k-means", "hierarchical"}, optional
            Clustering method to use. Default is "k-means".
        plotStdMask : bool, optional
            Whether to display a mask of the high standard deviation regions.
        plotScree : bool, optional
            Whether to display a scree plot of PCA variance explained.
        plotClusterMap : bool, optional
            Whether to display a 2D color-coded cluster map.
        plot3dClusterMap : bool, optional
            Whether to display a 3D scatter plot of the clustered data in PCA space.
        filter_size : int or None, optional
            Size of the median/majority filter to smooth the cluster map.
            If None, filtering is disabled.
        cluster_cmap : str or None, optional
            Name of the colormap to use for visualizing clusters.
            If None, uses 'gnuplot'.
        filter_iterations : int or None, optional
            Number of iterations to apply the majority filter. If None, disables filtering.
        outer_ring : int or None, optional
            Radius of the outer boundary for masking. If None, no outer ring mask is applied.
        polar : bool, optional
            Whether to apply polar coordinate-based masking.
        split_disconnected : bool, optional
            If True, post-process the 2D cluster map so that spatially disconnected
            islands within the same cluster label are split into separate clusters.
            This is done via `split_disconnected_clusters`, which relabels each
            connected component (using 8-connectivity) to a unique label.
            Note that this can increase the effective number of clusters beyond
            the original `n_clusters`.
    
        Returns
        -------
        cluster_map : numpy.ndarray
            A 2D array of shape `(A, B)` representing the cluster labels for each
            pixel in the original dataset. If `split_disconnected=True`, each
            spatially disconnected island is given a distinct label.
    
        Raises
        ------
        AssertionError
            If the input dataset is not 4-dimensional.
    
        Notes
        -----
        - The dataset is first masked to remove the center beam and optionally an
          outer ring (if `polar=True`).
        - Pixels with low standard deviation (below `std_Threshold`) are further
          masked out.
        - PCA is performed on the masked dataset to reduce its dimensionality to
          `n_PCAcomponents`.
        - Clustering is performed using the selected `clustering_method`.
        - Optionally, disconnected islands of a given label can be split into
          separate clusters via `split_disconnected_clusters`.
        - Smoothing of the cluster map can be achieved using a majority filter.
        """
        assert len(self.shape) == 4, "'dataset' must be 4-dimensional"
    
        # Remove center beam
        A, B, C, D = self.shape
    
        if not polar:
            dataset_noCenter = self.apply_mask(r_inner=r_centerBeam, r_outer=outer_ring).array
        else:
            dataset_noCenter = np.zeros_like(self.array)
            if outer_ring is not None:
                dataset_noCenter[:, :, r_centerBeam:outer_ring] = self.array[:, :, r_centerBeam:outer_ring]
            else:
                dataset_noCenter[:, :, r_centerBeam:] = self.array[:, :, r_centerBeam:]
    
        # Find pixels of high variation
        dataset_stdev = HyperData(dataset_noCenter ** power).get_stdDev(domain='reciprocal').array
    
        # Mask low standard deviation pixels
        low_std_dev_mask = dataset_stdev < std_Threshold * np.max(dataset_stdev)
        if plotStdMask:
            plt.figure()
            plt.imshow(low_std_dev_mask)
            plt.title(f'High Std. Dev. Mask, std_threshold = {std_Threshold}')
            plt.axis('off')
            plt.show()
    
        dataset_noCenter[:, :, low_std_dev_mask] = 0
        dataset_noCenter = dataset_noCenter.reshape(-1, C * D)
    
        # PCA for dimensionality reduction
        pca = PCA(n_components=n_PCAcomponents)
        data_reduced = pca.fit_transform(np.log(clip_values(dataset_noCenter)))
    
        if plotScree:
            plt.figure()
            plt.plot(range(1, n_PCAcomponents + 1), pca.explained_variance_ratio_, marker='o')
            plt.title("Scree Plot")
            plt.xlabel("Principal Component")
            plt.ylabel("Variance Explained")
            plt.show()
    
        # Clustering methods
        if clustering_method == "k-means":
            clustering_model = KMeans(n_clusters=n_clusters, random_state=0)
            clusters = clustering_model.fit_predict(data_reduced)
        elif clustering_method == "hierarchical":
            from scipy.cluster.hierarchy import linkage, fcluster
            linkage_matrix = linkage(data_reduced, method='ward')
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
    
        # Base colormap (may be updated after splitting)
        if cluster_cmap is not None:
            colormap = plt.cm.get_cmap(cluster_cmap, n_clusters)
        else:
            colormap = plt.cm.get_cmap('gnuplot', n_clusters)
    
        # Optional 3D plot (uses original cluster labels)
        if plot3dClusterMap:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(n_clusters):
                color = colormap(i)
                ax.scatter(
                    data_reduced[clusters == i, 0],
                    data_reduced[clusters == i, 1],
                    data_reduced[clusters == i, 2],
                    c=[color], label=f'Cluster {i + 1}', s=2, alpha=0.5
                )
            plt.show()
    
        # Map clusters back to the original dimensions
        cluster_map = clusters.reshape(A, B)
    
        # Optional majority / median filter
        if filter_size is not None and filter_iterations is not None:
            for _ in range(filter_iterations):
                cluster_map = median_filter(cluster_map, (filter_size, filter_size))
    
        # Optional: split disconnected islands into separate cluster labels
        if split_disconnected:
            # background set to a label that does not appear (e.g., -1)
            # here we treat all labels as foreground; background=-1 simply does nothing special
            cluster_map, mapping = split_disconnected_clusters(
                cluster_map,
                connectivity=1,
                background=-1
            )
            # Update effective number of clusters and colormap
            n_clusters = int(cluster_map.max()) + 1
            if cluster_cmap is not None:
                colormap = plt.cm.get_cmap(cluster_cmap, n_clusters)
            else:
                colormap = plt.cm.get_cmap('gnuplot', n_clusters)
    
        # Plot the 2D cluster map
        if plotClusterMap:
            cluster_map_colored = np.zeros((A, B, 3), dtype=np.uint8)
            for i in range(n_clusters):
                color = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
                indices = np.where(cluster_map == i)
                cluster_map_colored[indices] = color
    
            plt.figure()
            plt.imshow(cluster_map_colored)
            plt.title(f"Cluster Map ({A}x{B}) with {n_clusters} Clusters")
            plt.axis('off')
            plt.show()
    
        return cluster_map


    #TODO: add option to automatically remove the per-cluster backgrounds by adding 
    #      a parameter `cluster_map` that will have the same shape as the real-space 
    def remove_bg(self, background, bg_frac=1, residual_bg_frac=0, **resBg_kwargs):
        """
        Subtracts a fraction of the background from the dataset and clips the 
        result to handle underflows.
        
        Parameters
        ----------
        background : ndarray
            The background data array which must be of the same shape as self.array.
        bg_frac : float, optional
            The fraction of the background to be subtracted from the dataset. 
            Must be between 0 and 1 (inclusive).
        
        Returns
        -------
        HyperData
            A new instance of HyperData with the background subtracted.
        
        Raises
        ------
        ValueError
            If 'bg_frac' is not within the required range [0, 1] or 'background' is 
            not of the same shape as the diffraction patterns.
        """
        if not (0 <= bg_frac <= 1):
            raise ValueError("'bg_frac' must be between 0 and 1, inclusive.")
        
        if not (0 <= residual_bg_frac <= 1):
            raise ValueError("'residual_bg_frac' must be between 0 and 1, inclusive.")
        
        if background.shape != self.array.shape[-2:]:
            raise ValueError("""'background' must match the shape of the last 
                             two dimensions of the diffraction dataset.""")
        
        if type(background) != np.ndarray:
            background = background.array
        
        if residual_bg_frac > 0:
            residual_bg = self.get_residualBg(**resBg_kwargs)
            return HyperData(self.array[:,:] - background*bg_frac - residual_bg*residual_bg_frac).clip()
        else:
            return HyperData(self.array[:,:] - background*bg_frac).clip()

    #TODO: remove the `center` parameter by automatically getting the CoM of the DP
    def get_polarTransform(self,
                           center: Tuple[float, float] = None,
                           r_max: int = None,
                           output_shape: Tuple[int, int] = None,
                           order: int = 1
                           ) -> Tuple["HyperData", int]:
        """
        Return a new HyperData object where each diffraction pattern slice
        has been remapped from Cartesian to polar. Uses fast interpolation
        via scipy.ndimage.map_coordinates and tqdm for progress.

        Parameters
        ----------
        center : tuple of two floats, optional
            (y_center, x_center) in pixel coordinates. Defaults to the image midpoint.
        r_max : int, optional
            Maximum radius (in pixels) to sample. Defaults to min(height, width)//2.
            If r_max > min(center), those regions will be filled with 1 (instead of NaN).
        output_shape : tuple (n_r, n_theta), optional
            Desired shape of the output polar image. 
            - n_r = number of radial samples
            - n_theta = number of angular samples
            Default: (r_max, r_max) so that the result is square.
        order : int, default=1
            The spline interpolation order for map_coordinates (0=nearest,
            1=bilinear, 3=cubic, etc.).

        Returns
        -------
        (polar_hd, r_max_used) : tuple
            - polar_hd: HyperData instance whose `.array` has shape either
                         (B, n_r, n_theta) or (A, B, n_r, n_theta),
                         matching the original dimensionality but polar-transformed.
            - r_max_used: the integer radius actually used for this transform.
        """
        arr = self.array
        shp = arr.shape

        # 1) Determine input dims and default center/r_max
        if len(shp) == 4:
            A, B, C, D = shp
            height, width = C, D
        elif len(shp) == 3:
            A = None
            B, C, D = shp
            height, width = C, D
        else:
            raise ValueError("HyperData must be 3D (B×C×D) or 4D (A×B×C×D).")

        # Default center at image midpoint
        cy = (height - 1) / 2.0
        cx = (width  - 1) / 2.0
        if center is not None:
            cy, cx = center

        # Default r_max = floor(min(height, width)//2)
        default_rmax = int(min(height, width) // 2)
        if r_max is None:
            r_max_used = default_rmax
        else:
            r_max_used = int(r_max)

        # 2) Determine output_shape = (n_r, n_theta)
        if output_shape is None:
            n_r     = r_max_used
            n_theta = r_max_used
        else:
            n_r, n_theta = output_shape

        # 3) Precompute the polar→Cartesian mapping grid
        #    Radial values from 0 to r_max_used in n_r steps
        r_vals = np.linspace(0, r_max_used, n_r)
        #    Theta from -π to +π in n_theta steps
        theta_vals = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

        #    Meshgrid in (r, θ), shape = (n_r, n_theta) when indexing='ij'
        #    But map_coordinates expects coords stacked as [row_coords; col_coords].
        theta_grid, r_grid = np.meshgrid(theta_vals, r_vals, indexing='xy')
        #    Convert to Cartesian (floating) coordinates
        x_grid = cx + r_grid * np.cos(theta_grid)  # shape: (n_r, n_theta)
        y_grid = cy + r_grid * np.sin(theta_grid)  # shape: (n_r, n_theta)

        #    Stack into a 2×(n_r·n_theta) array for map_coordinates
        coords = np.vstack((
            y_grid.ravel(),  # row indices
            x_grid.ravel()   # col indices
        ))

        # 4) Prepare an output array of the correct shape
        if A is not None:
            out_arr = np.zeros((A, B, n_r, n_theta), dtype=arr.dtype)
        else:
            out_arr = np.zeros((B, n_r, n_theta), dtype=arr.dtype)

        # 5) Define a helper mask for “blank” regions: whenever r_grid > min(cy, cx),
        #    we’ll force those polar pixels to 1.0 after interpolation.
        r_limit = min(cy, cx)
        blank_mask = (r_grid > r_limit).ravel()  # shape: (n_r*n_theta,)

        # 6) Loop over all slices with tqdm progress bars
        if A is not None:
            outer_desc = "Diffraction patterns"
            inner_desc = "Diffraction slice"
            for i in tqdm(range(A), desc=outer_desc):
                for j in tqdm(range(B), desc=inner_desc, leave=False):
                    diff = arr[i, j]
                    # Interpolate the 2D slice onto our polar grid
                    polar_flat = map_coordinates(
                        diff,
                        coords,
                        order=order,
                        mode='constant',
                        cval=0.0
                    )
                    # Replace blank‐region entries (where r > r_limit) with 1
                    polar_flat[blank_mask] = 1.0

                    # Reshape back to (n_r, n_theta) and clip
                    polar_img = polar_flat.reshape((n_r, n_theta))
                    out_arr[i, j] = clip_values(polar_img)
        else:
            desc = "Diffraction slice"
            for j in tqdm(range(B), desc=desc):
                diff = arr[j]
                polar_flat = map_coordinates(
                    diff,
                    coords,
                    order=order,
                    mode='constant',
                    cval=0.0
                )
                polar_flat[blank_mask] = 1.0
                polar_img = polar_flat.reshape((n_r, n_theta))
                out_arr[j] = clip_values(polar_img)

        # 7) Return the new HyperData and the r_max_used
        return HyperData(out_arr)

    def get_average_clusters(self,
                             cluster_map: np.ndarray,
                             domain: str = 'real',
                             plot_averages: bool = True,
                             vmin: float = 4,
                             vmax: float = 14,
                             cmap: str = 'turbo',
                             logScale: bool = True) -> "HyperData":
        """
        Average over clusters defined in either the real (scan) domain or
        the reciprocal (diffraction) domain.

        Parameters
        ----------
        cluster_map : 2D array of ints
            If domain=='real', shape must equal (A, B).
            If domain=='reciprocal', shape must equal (C, D).
        domain : {'real', 'reciprocal'}
            Which axes of `self.array` the cluster_map indexes:
              - 'real'      → cluster_map shape = (A, B)
              - 'reciprocal'→ cluster_map shape = (C, D)
        plot_averages : bool
            If True, show each cluster’s average as an image.
        vmin, vmax, cmap : passed to plt.imshow
        logScale : bool
            If True, plot np.log(average + 1) instead of raw average.

        Returns
        -------
        HyperData
            Holds an array of shape
              - (n_clusters, C, D) for domain='real'
              - (n_clusters, A, B) for domain='reciprocal'
        """
        arr = self.array
        shp = arr.shape

        # must be 4D
        if arr.ndim != 4:
            raise ValueError("`get_average_clusters` only supports 4D HyperData")

        A, B, C, D = shp
        E, F = cluster_map.shape

        # pick domain
        if domain == 'real':
            if (E, F) != (A, B):
                raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ data real‐domain {(A,B)}")
            # cluster labels and counts
            labels = np.unique(cluster_map)
            n_clusters = labels.size
            # output: average diffraction per real‐domain cluster
            avg = np.zeros((n_clusters, C, D), dtype=arr.dtype)

            for idx, lbl in enumerate(labels):
                mask = (cluster_map == lbl)
                # arr[mask, :, :] shapes to (#pixels_in_cluster, C, D)
                avg[idx] = arr[mask, :, :].mean(axis=0)

        elif domain == 'reciprocal':
            if (E, F) != (C, D):
                raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ data reciprocal‐domain {(C,D)}")
            labels = np.unique(cluster_map)
            n_clusters = labels.size
            # output: average real‐space image per reciprocal‐domain cluster
            avg = np.zeros((n_clusters, A, B), dtype=arr.dtype)

            for idx, lbl in enumerate(labels):
                mask = (cluster_map == lbl)
                # arr[:, :, mask] → shape (A, B, #pixels)
                avg[idx] = arr[:, :, mask].mean(axis=2)

        else:
            raise ValueError("`domain` must be 'real' or 'reciprocal'")

        # plotting
        if plot_averages:
            for i in range(n_clusters):
                data = avg[i] + 1
                if logScale:
                    disp = np.log(data)
                else:
                    disp = data

                plt.figure(dpi=150)
                plt.imshow(disp, vmin=vmin, vmax=vmax, cmap=cmap,)
                plt.axis('off')
                plt.title(f"Cluster {i} ({domain})")
                plt.show()

        return HyperData(avg)

    def get_stdev_clusters(self,
                           cluster_map: np.ndarray,
                           threshold: float     = None,
                           r_min: float         = 0.0,
                           r_max: float         = None,
                           logScale: bool       = False,
                           power: float         = None
                          ) -> "HyperData":
        """
        Compute per‐cluster standard deviations in the diffraction domain
        (real‐space clustering only), with optional preprocessing
        (power‐law or log transform), and optional thresholding
        between radii [r_min, r_max].

        Parameters
        ----------
        cluster_map : 2D int array, shape (A, B)
            Real‐space cluster labels.
        threshold : float in [0,1], optional
            If None, returns raw std‐dev arrays.
            If set, returns boolean masks where
              std >= threshold * max(std within [r_min,r_max]).
        r_min : float, default=0.0
            Inner radius (px) to exclude (previously `r_center`).
        r_max : float, optional
            Outer radius (px) to exclude beyond this in thresholding.
            Defaults to `min(cy, cx)` (the largest fully‐inside radius).
        logScale : bool, default=False
            If True, apply np.log(data + 1) before computing std‐dev.
        power : float, optional
            If provided, raise each pattern to this power before any log.

        Returns
        -------
        HyperData
            If `threshold is None`:  `.array` shape = (n_clusters, C, D) of float std‐devs.
            If `threshold` set:      `.array` shape = (n_clusters, C, D) of bool masks.
        """
        arr = self.array
        if arr.ndim != 4:
            raise ValueError("get_stdev_clusters only supports 4D HyperData")
        A, B, C, D = arr.shape

        if cluster_map.shape != (A, B):
            raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ (A,B)=({A},{B})")

        # 1) compute raw std‐dev per cluster
        labels = np.unique(cluster_map)
        n_clusters = labels.size
        stdev_arr = np.zeros((n_clusters, C, D), dtype=float)

        for i, lbl in enumerate(labels):
            sel = (cluster_map == lbl)
            data = arr[sel]  # shape = (#pixels_in_cluster, C, D)
            if power is not None:
                data = data ** power
            if logScale:
                data = np.log(data + 1)
            stdev_arr[i] = data.std(axis=0) if data.size else 0.0

        # 2) if no threshold → return the raw std‐dev map
        if threshold is None:
            return HyperData(stdev_arr)

        # 3) validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("`threshold` must be between 0 and 1")

        # 4) build radial masks
        y, x = np.indices((C, D))
        cy, cx = (C - 1) / 2.0, (D - 1) / 2.0
        dist = np.hypot(y - cy, x - cx)

        # default r_max → largest fully‐inside radius
        r_max_used = r_max if r_max is not None else min(cy, cx)

        # we only consider pixels with r_min ≤ r ≤ r_max_used
        region = (dist >= r_min) & (dist <= r_max_used)

        # 5) build boolean mask per cluster
        bool_arr = np.zeros_like(stdev_arr, dtype=bool)
        for i in range(n_clusters):
            sd = stdev_arr[i]
            valid = region
            max_sd = sd[valid].max() if np.any(valid) else 0.0
            thresh_val = threshold * max_sd
            bool_arr[i] = (sd >= thresh_val) & region

        return HyperData(bool_arr)
    
#%%

#TODO: figure out if we can define the scale of the pixel somehow `A^-1` or `mrad`
class ReciprocalSpace:

    def __init__(self, data):
        self.array = data
        self.shape = data.shape
        self.denoiser = Denoiser(data)
    
    def show(self,
             power: float = 1,
             title: str = 'Diffraction Pattern',
             logScale: bool = True,
             axes: bool = True,
             vmin=None,
             vmax=None,
             figsize=(10, 10),
             aspect=None,
             cmap: str = 'turbo',
             coords: np.ndarray | None = None,
             **scatter_kwargs):
        """
        Visualize the diffraction pattern stored in this ReciprocalSpace object.
    
        The diffraction pattern is displayed with optional log-scaling and
        intensity exponentiation. Optionally, a set of (y, x) peak coordinates
        can be overlaid as a scatter plot on top of the image.
    
        Parameters
        ----------
        power : float, optional
            Intensity exponent. If `logScale` is True, the displayed image is
            `power * log(self.array)`. If `logScale` is False, the displayed
            image is `self.array ** power`. Default is 1.
        title : str, optional
            Title for the diffraction pattern (used on the axes when `axes=True`).
        logScale : bool, optional
            If True, use a logarithmic transform (`log`) of the data before
            exponentiation. If False, use a pure power-law transform.
        axes : bool, optional
            If True, show axes, ticks, and a colorbar. If False, hide axes and
            only show the image.
        vmin : float or None, optional
            Minimum intensity for color scaling. If None, it is inferred from
            the transformed data.
        vmax : float or None, optional
            Maximum intensity for color scaling. If None, it is inferred from
            the transformed data.
        figsize : tuple, optional
            Figure size passed to `plt.figure(figsize=...)`.
        aspect : float, str, or None, optional
            Aspect ratio for the displayed image. If not None, passed to
            `ax.set_aspect(aspect)`.
        cmap : str, optional
            Matplotlib colormap name for the image. Default is 'turbo'.
        coords : array-like of shape (N, 2), optional
            Optional array of peak coordinates to overlay as scatter points.
            Each row should be `[y, x]`. When provided, points are plotted at
            `x = coords[:, 1]` and `y = coords[:, 0]` on top of the image.
        **scatter_kwargs :
            Additional keyword arguments forwarded to `plt.scatter(...)` for
            the overlay points (e.g., `c='r'`, `s=20`, `marker='x'`, etc.).
    
        Notes
        -----
        - The method assumes `self.array` is a 2D diffraction pattern.
        - For `logScale=True`, non-positive values in `self.array` will produce
          `-inf` or `nan` in the log; it is recommended to use background-
          subtracted and strictly positive data when using log scaling.
        """
    
        # Visualize
        plt.figure(figsize=figsize)
    
        if logScale:
            processed_data = power * np.log(self.array)
        else:
            processed_data = self.array ** power
    
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.min(processed_data)
        if vmax is None:
            vmax = np.max(processed_data)
    
        # Display the image with the specified color mapping and value limits
        im1 = plt.imshow(processed_data, vmin=vmin, vmax=vmax, cmap=cmap)
    
        ax = plt.gca()
    
        # Aspect ratio
        if aspect is not None:
            ax.set_aspect(aspect)
    
        # Optional scatter overlay of coordinates
        if coords is not None:
            coords = np.asarray(coords)
            if coords.size > 0:
                plt.scatter(coords[:, 1], coords[:, 0], **scatter_kwargs)
    
        if axes:
            ky, kx = self.shape
    
            plt.axis('on')
            plt.xticks(np.arange(0, kx, 5))
            plt.yticks(np.arange(0, ky, 5))
    
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
    
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            ax.set_title(title, fontsize=18)
    
            if power:
                cbar_title = f"Intensity\n(Power = {power})"
            else:
                cbar_title = "Intensity"
    
            plt.title(cbar_title, fontsize=14)
        else:
            plt.axis('off')
    
        plt.show()

    
    #TODO: how to call method from HyperData?
    # def remove_center_beam(self):
    #     """
        
    #     """
    #     return ReciprocalSpace(annular)
    
    def crop(self, kylim=None, kxlim=None, kshape=None):
        """
        Crop a diffraction pattern (ReciprocalSpace object) with optional
        subpixel precision and enforced resizing.
    
        Parameters
        ----------
        kylim : int, float, or tuple, optional
            Vertical reciprocal-space limits. Can be float for subpixel cropping.
        kxlim : int, float, or tuple, optional
            Horizontal reciprocal-space limits. Can be float for subpixel cropping.
        kshape : tuple of (int, int), optional
            Output shape (A, B) for resizing the cropped pattern.
            If not provided, inferred from crop size.
    
        Returns
        -------
        ReciprocalSpace
            New object containing the cropped (and optionally resized) diffraction pattern.
        """
    
        def parse_limits(limits, max_length):
            if limits is None:
                return (0, max_length)
            elif isinstance(limits, (int, float)):
                if limits < 0 or limits > max_length:
                    raise ValueError("Index out of bounds")
                return (limits, limits + 1)
            elif isinstance(limits, tuple):
                start, end = limits
                if start < 0 or end > max_length:
                    raise ValueError("Invalid range or out of bounds")
                return (start, end)
            else:
                raise ValueError("Limits must be int, float, tuple, or None")
    
        # --- Parse reciprocal-space limits ---
        ky, kx = self.shape
        kylim_range = parse_limits(kylim, ky)
        kxlim_range = parse_limits(kxlim, kx)
    
        # Integer boundaries for extraction
        y0, y1 = int(np.floor(kylim_range[0])), int(np.ceil(kylim_range[1]))
        x0, x1 = int(np.floor(kxlim_range[0])), int(np.ceil(kxlim_range[1]))
    
        # Detect if subpixel cropping is requested
        subpixel = not all(isinstance(v, int) for v in kylim_range + kxlim_range)
    
        # Default output shape
        if kshape is None:
            A = int(round(kylim_range[1] - kylim_range[0]))
            B = int(round(kxlim_range[1] - kxlim_range[0]))
            kshape = (A, B)
    
        if kshape[0] != kshape[1]:
            print(f"Warning: Non-square output shape {kshape}. Proceeding anyway.")
    
        # --- Extract raw region ---
        cropped = self.array[y0:y1, x0:x1]
    
        # --- Handle subpixel or resize ---
        if subpixel or kshape is not None:
            cropped = transform.resize(cropped, kshape,
                             order=1,  # bilinear
                             mode='reflect',
                             anti_aliasing=True)
    
        return ReciprocalSpace(cropped)
    
    def get_spotCenter(self, ky, kx, r, method='CoM', plotSpot=False,):
        """
        Find the center of mass (of pixel intensities) of a diffraction spot, 
        allowing for non-integer radii.
        """
        
        # Pad and round up to ensure the entire radius is accommodated
        pad_width = int(np.ceil(r))
        padded_data = np.pad(self.array, pad_width=pad_width, mode='constant')
    
        # Adjustment of padding coordinates
        ky_padded, kx_padded = ky + pad_width, kx + pad_width
        
        # Determine the size of the area to extract based on r, ensuring it matches the mask's dimensions
        area_size = int(np.ceil(r * 2))
        # if area_size % 2 == 0:
        #     area_size += 1  # Ensure the area size is odd to match an odd-sized mask
            
        # Generate the circular mask with the correct dimensions
        mask = circular_mask((area_size) // 2, area_size // 2, r)
        
        # Extract the region of interest from the padded data
        ymin, ymax = int(ky_padded - (area_size // 2)), int(ky_padded + (area_size // 2)) + 1
        xmin, xmax = int(kx_padded - (area_size // 2)), int(kx_padded + (area_size // 2)) + 1
        spot_data = padded_data[ymin:ymax, xmin:xmax]
    
        # Check if shapes match, otherwise adjust
        if spot_data.shape != mask.shape:
            min_dim = min(spot_data.shape[0], mask.shape[0], spot_data.shape[1], mask.shape[1])
            spot_data = spot_data[:min_dim, :min_dim]
            mask = mask[:min_dim, :min_dim]
    
        # Apply mask and calculate its CoM
        masked_spot_data = spot_data * mask
        
        # Find peak maximum using the chosen method
        if method == 'CoM':
            com_y, com_x = center_of_mass(masked_spot_data)
            
        elif method == 'gaussian':
            com_x, com_y = fit_gaussian_2d(masked_spot_data)
            
        elif method == 'elliptical_gaussian':
            com_x, com_y = fit_gaussian_2d(masked_spot_data)    
        
        if plotSpot:
            # Create turbo colormap with 0-values white
            base_cmap = plt.cm.turbo
            custom_cmap = ListedColormap(np.concatenate(([np.array([1, 1, 1, 1])], 
                                                         base_cmap(np.linspace(0, 1, 2**12))[1:]), axis=0))
            plt.imshow(masked_spot_data, cmap=custom_cmap)
            plt.colorbar()
            plt.scatter(com_x, com_y, color='yellow', s=50, label='Subpixel CoM')
            plt.show()
        
        ky_padded = com_y + ymin
        kx_padded = com_x + xmin
                
        # Adjust the CoM to account for padding
        ky_CoM = ky_padded - pad_width
        kx_CoM = kx_padded - pad_width
    
        return ky_CoM, kx_CoM
    
    def get_centers(self, r, ref_coords, show=False, method='CoM'):
        """
        Generate an array spot centers for any DP
        
        r : int, float or list
            if list, its length must be the same as that of coords
        """
        
        assert len(self.shape) == 2, "Input data must be of 2-dimensional"
        
        num_peaks = len(ref_coords)
        
        centers = np.zeros((num_peaks, 2))
        for j in range(num_peaks):
            
            if isinstance(r, (np.ndarray, list)):
                r = r[j]
                
            centers[j] = self.get_spotCenter(ref_coords[j, 0], ref_coords[j, 1],
                                            r + 1e-10,  method, show,)
                                       
        return centers

    #TODO: enable functionality for 4-fold symmetry as well
    #TODO: automatically find the number of peaks on each order by looking at the 
    #      distance from center beam (frequency)
    def masked_DPs(self, mask_radius, centers=None, ref_coords=None, order=None, title=None, 
                   return_mask=False, plot=True, method='CoM'):
        """ 
        Generate masked diffraction plots for each order.
        Order = 1,2,3,4; Last option plots all.
        """
        
        if centers is None and ref_coords is None:
            raise ValueError("Either 'centers' is None or 'ref_coords' is None but not both.")
        
        A,B = self.shape
        compound_mask = np.zeros((A, B))
        
        if centers is None:
            centers = self.get_centers(r=mask_radius, ref_coords=ref_coords, method=method)
    
        for mask_center in centers:
                    
            mask_spot = make_mask(mask_center, mask_radius, mask_dim=(A,B))
            compound_mask = compound_mask + mask_spot     
        
        # Apply compound mask
        masked_data = self.array*compound_mask
        
        if plot: 
            
            plt.figure(figsize=(10, 10))
            base_cmap = plt.cm.turbo
        
            # Create a new colormap from the existing colormap
            # np.concatenate combines the arrays. The first array is just [1, 1, 1, 1] which corresponds to white in RGBA.
            # We take the colormap 'turbo', convert it to an array, and exclude the first color to make room for white.
            custom_cmap = ListedColormap(np.concatenate(([np.array([1, 1, 1, 1])], 
                                                         base_cmap(np.linspace(0, 1, 2**12))[1:]), axis=0))
        
            # Plotting 
            im1 = plt.imshow(masked_data, cmap=custom_cmap)
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            # Title settings based on 'spots_order'
            if title:
                assert type(title) == str, "The title of the plot must be a string."
                ax.set_title(title, fontsize=18)
            else:
                ax.set_title('Masked Diffraction Pattern', fontsize=18)
            
            # Colorbar settings
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            plt.title("Intensity", fontsize=14, pad=20)  # Adjusted to avoid overlapping with the main title
        
            plt.show()
        
        if return_mask:
            return compound_mask
        
    def get_intensities(self, 
                        r,
                        centers=None,  
                        ref_coords=None, 
                        method='CoM',
                        compute_resBg=False,
                        residual_frac=0.9,
                        **resBg_kwargs):
        """
        Extract Bragg peak intensities from a single DP
        
        residual_pxBg is an integer, float, or 
        """
    
        if centers is None:
            centers = self.get_centers(r, ref_coords=ref_coords, method=method)
    
        ints = np.zeros(len(centers))
        
        for int_idx, intensity in enumerate(ints):
    
            if isinstance(r, (list, np.ndarray)):
                r = r[int_idx]        
    
            masked_data = self.array*make_mask(centers[int_idx], r_mask=r+1e-10, mask_dim=self.shape)
            ints[int_idx] = np.sum(masked_data[round(centers[int_idx][0]-(r+0.5)):round(centers[int_idx][0]+(r+0.5)),
                                                round(centers[int_idx][1]-(r+0.5)):round(centers[int_idx][1]+(r+0.5))])
        
        if compute_resBg:
            res_bg = self.get_residualBg(centers=centers, **resBg_kwargs)
            ints -= res_bg * (np.pi*r**2) * residual_frac
                
        return ints
    
    
    def get_residualBg(self, centers, r_spots, bg_method='rings', t_ring=None, show=False, **kwargs):
        """
        Calculate the residual background value around Bragg peaks.
    
        The function supports three methods to calculate the background:
        
        - The 'rings' method masks the immediate annular region around each Bragg peak 
          at the positions in 'centers' and returns the mean background value 
          surrounding each Bragg peak.
        - The 'rings_mean' method creates a ring around each Bragg peak and returns 
          the average background value per pixel for all the rings combined.
        - The 'grimms_ring' method creates a large annular mask that ideally passes 
          through the Bragg peaks located at 'centers'. It hollows out the circular 
          regions enclosing the Bragg peaks, so only the space between the peaks is 
          masked. This method assumes that the Bragg peaks at positions 'centers' fall 
          within a common annular region.
    
        Parameters
        ----------
        centers : array-like
            An array of shape (N, 2) containing the (ky, kx) positions of the Bragg peaks.
        r_spots : float or tuple of floats
            Radius of the spots to mask. If a tuple (inner_radius, outer_radius) 
            is provided, an annular mask is created.
        bg_method : str, optional
            Method to use for calculating the background. Options are 'rings', 
            'rings_mean', and 'grimms_ring'. Default is 'rings'.
        t_ring : float, optional
            Thickness of the ring for the 'grimms_ring' method. If not specified, 
            it is set to 1.5 times the radius of the spots.
        show : bool, optional
            If True, display the masked data. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the display function if `show` is True.
    
        Returns
        -------
        res_bgs : np.ndarray
            The mean background value surrounding each Bragg peak (for 'rings' 
            method) or the average background value per pixel (for 'rings_mean' 
            and 'grimms_ring' methods).
        """
        
        bg_methods = ['rings', 'rings_mean', 'grimms_ring']
        assert bg_method in bg_methods, f"Input 'method' must be one of the follwing: {bg_method}"
        
        A, B = self.shape
        dp_center = ((A-1)/2, (B-1)/2)
        
        if bg_method=='rings':
            if not isinstance(r_spots, tuple):
                raise ValueError(""""For the 'rings' method, the input parameter 
                                     'r_spots' must be a tuple specifying the inner 
                                     and outer radius of bg. region arounf each each spot""")
            
            res_bgs = np.zeros(len(centers))
            dp_mask = np.zeros_like(self.array, dtype=bool)
            # We collect multiple values for the intensities corresponding to each Bragg peak
            for c_idx, center in enumerate(centers):
                bool_mask = make_mask(center, r_spots, mask_dim=(A,B))
                masked_dp = self.array * bool_mask
                dp_mask += bool_mask
                res_bgs[c_idx] = np.sum(masked_dp)/np.sum(bool_mask)
            if show:
                ReciprocalSpace(self.array * dp_mask).show(**kwargs)
            return res_bgs
     
        if bg_method=='rings_mean':            
            if not isinstance(r_spots, tuple):
                raise ValueError(""""For the 'rings_mean' method, the input parameter 
                                     'r_spots' must be a tuple specifying the inner 
                                     and outer radius of bg. region arounf each each spot""")
            # In this case, a ring is drawn araound every spot
            bool_mask = make_mask(centers, r_spots, mask_dim=(A,B))
        
        if bg_method=='grimms_ring':
            if isinstance(r_spots, tuple):
                raise ValueError(""""For the 'grimms_ring' method, the input parameter 
                                     'r_spots' must be an integer or float specifying 
                                     the radius to blank each spot""")
            
            # If not defined, we define a ring thickness equal to 1.5 times the radii of the enclosed Bragg peaks
            # Note that larger ring thickness may result in inaccurate results due to enclosing signal from other
            # Bragg peaks
            if isinstance(r_spots, (list, np.ndarray)):
                r_mean = np.mean(np.array(r_spots))
            else:
                r_mean = r_spots
            if t_ring is None:
                t_ring = 1.5*r_mean
            elif not isinstance(t_ring, (int, float)):
                raise ValueError("'t_ring' must be an integer or float.")
            elif t_ring > 3*r_mean or t_ring < 1.5:
                raise ValueError("""The input parameter t_ring must be within the 
                                    range [1.5, 3*r_spots]""")
            
            # We automatically define the mean radius of the ring based on Bragg peak positions
            distances = np.sqrt((centers[:, 0] - dp_center[0]) ** 2 + (centers[:, 1] - dp_center[1]) ** 2)
            r_ring = np.mean(distances)
            
            # We combine the masks enclosing the Bragg peaks and the annular region covering them
            spots_mask = make_mask(centers, r_mean, mask_dim=(A,B), invert=True)
            annular_mask = make_mask(dp_center, (r_ring - t_ring/2, r_ring + t_ring/2))
            
            bool_mask = np.logical_and(spots_mask, annular_mask)
        
        # Apply mask
        masked_dp = self.array * bool_mask
        
        if show:
            ReciprocalSpace(masked_dp).show(**kwargs)
        
        # Return the average background value per pixel
        return np.sum(masked_dp)/np.sum(bool_mask)
    
    #TODO: if n_fold is used, then the peaks should be returned in groups of n, all separated by 360/n and being the same distance from the center
    def get_peaks(self,
                  radius: float,
                  min_distance: int,
                  trench_width: float = 1.0,
                  kernel_amp: float = 1.0,
                  trench_amp: float = -0.5,
                  threshold_abs: float = 1.0,
                  threshold_rel: float = None,
                  r_range: tuple = None,
                  n_fold: int | None = None,
                  sym_mode: str = "none",
                  sym_tolerance_px: float = 2.0,
                  center_tolerance_px: float = 2.0,
                  orbit_min_fraction: float = 0.5,
                  reorder: bool = False) -> np.ndarray:
        """
        Detect peaks in a 2D diffraction pattern via template matching with a
        disk and a negative "trench" kernel, optionally restricted to an
        annular region [r_min, r_max] (to exclude, for example, the central
        beam and far-away, low-signal peaks).
    
        Optionally, enforce n-fold rotational symmetry about the pattern center
        over the full Bragg peak array (recommended for centrosymmetric
        diffraction patterns with approximate n-fold symmetry).
    
        Notes
        -----
        - It is recommended to use background-subtracted data and to have all
          diffraction patterns center-beam aligned.
        - When using n_fold symmetry, it is generally recommended to have a
          reasonably clean peak detection (thresholds, min_distance) so that
          orbits can be reliably inferred.
    
        Parameters
        ----------
        radius : float
            Radius of the positive disk (in pixels).
        trench_width : float, optional
            Width of the negative surround ring (in pixels).
        kernel_amp : float, optional
            Amplitude of the central disk.
        trench_amp : float, optional
            Amplitude (negative) of the surrounding trench.
        threshold_abs : float, optional
            Absolute correlation threshold for peak detection.
        threshold_rel : float, optional
            Relative threshold (fraction of max correlation) if threshold_abs
            is None. If r_range is provided, the maximum is computed only
            inside the annulus.
        min_distance : int, optional
            Minimum number of pixels separating peaks (for suppression).
        r_range : tuple (r_min, r_max), optional
            If provided, only search for peaks whose radial distance from the
            dp center lies within [r_min, r_max] (in pixels).
        n_fold : int or None, optional
            Order of rotational symmetry to enforce (e.g., 4 for 4-fold, 6 for
            6-fold). If None or < 2, no symmetry enforcement is applied.
        sym_mode : {"none", "repair", "prune", "both"}, optional
            How to enforce n_fold symmetry on the detected peaks:
            - "none":   no symmetry post-processing.
            - "repair": keep all detected peaks and add missing symmetric ones
                        for sufficiently complete orbits.
            - "prune":  keep only peaks belonging to sufficiently complete
                        n-fold orbits; do not add synthetic peaks.
            - "both":   keep only peaks in sufficiently complete orbits and
                        add synthetic peaks at missing symmetric positions.
        sym_tolerance_px : float, optional
            Maximum Euclidean distance (in pixels) between a detected peak and
            its ideal symmetric position for them to be considered the same.
            Also used to merge peaks that end up too close after “repair”.
        center_tolerance_px : float, optional
            Radial tolerance (in pixels) used to classify peaks as belonging
            to the central beam region, which is excluded from symmetry
            grouping and passed through unchanged (after de-duplication).
        orbit_min_fraction : float, optional
            Minimum fraction of the n_fold orbit that must be present in the
            detected peaks for that orbit to be considered "real". Orbits
            below this fraction are treated as noise and ignored in "prune"
            and "both" modes (and will not be repaired in "repair" mode).
    
        Returns
        -------
        coords : (M, 2) np.ndarray
            Array of (y, x) coordinates of detected peaks (possibly augmented
            and/or pruned by symmetry enforcement).
    
        by Adan J. Mireles
        Applied Physics Graduate Program, Rice University
    
        July 2025
        """
    
        def _merge_close_points(arr: np.ndarray, tol: float) -> np.ndarray:
            """
            Merge points closer than tol in Euclidean distance.
            Keeps the first occurrence and discards later ones within tol.
            """
            arr = np.asarray(arr, float)
            if arr.size == 0:
                return arr.reshape(0, 2)
    
            kept = []
            for p in arr:
                if not kept:
                    kept.append(p)
                    continue
                kp = np.asarray(kept)
                d2 = (kp[:, 0] - p[0]) ** 2 + (kp[:, 1] - p[1]) ** 2
                if np.all(d2 > tol * tol):
                    kept.append(p)
            return np.asarray(kept)
    
        def _finalize_coords(arr: np.ndarray, tol: float) -> np.ndarray:
            """
            Merge close points, round to int, and drop exact duplicates.
            """
            if arr.size == 0:
                return arr.reshape(0, 2).astype(int)
            merged = _merge_close_points(arr, tol)
            arr_int = np.round(merged).astype(int)
            arr_int = np.unique(arr_int, axis=0)
            return arr_int
    
        def _enforce_nfold_symmetry(coords: np.ndarray,
                                    Cy: float,
                                    Cx: float,
                                    n_fold: int | None,
                                    sym_mode: str,
                                    sym_tolerance_px: float,
                                    center_tolerance_px: float,
                                    orbit_min_fraction: float) -> np.ndarray:
            """
            Enforce n-fold rotational symmetry on a set of peak coordinates.
            """
            if n_fold is None or n_fold < 2:
                return coords
            if sym_mode == "none":
                return coords
            if coords.size == 0:
                return coords
            if sym_mode not in ("none", "repair", "prune", "both"):
                raise ValueError("sym_mode must be one of: 'none', 'repair', 'prune', 'both'")
    
            coords = coords.astype(float)
            ys = coords[:, 0]
            xs = coords[:, 1]
    
            dy = ys - Cy
            dx = xs - Cx
            radii = np.hypot(dx, dy)
    
            # Separate central-beam peaks (passed through unchanged, after merging)
            center_mask = radii <= center_tolerance_px
            center_coords = coords[center_mask]
            ring_coords = coords[~center_mask]
    
            center_out = _finalize_coords(center_coords, center_tolerance_px)
    
            if ring_coords.size == 0:
                return center_out
    
            ys = ring_coords[:, 0]
            xs = ring_coords[:, 1]
            M = len(ring_coords)
    
            theta = 2.0 * np.pi / float(n_fold)
            tol2 = sym_tolerance_px ** 2
    
            def rotate_point(y, x, k):
                """Rotate (y, x) by k * theta about (Cy, Cx)."""
                dy_ = y - Cy
                dx_ = x - Cx
                ang = k * theta
                ca = np.cos(ang)
                sa = np.sin(ang)
                dy_r = ca * dy_ - sa * dx_
                dx_r = sa * dy_ + ca * dx_
                return Cy + dy_r, Cx + dx_r
    
            good_measured = []  # peaks belonging to sufficiently complete orbits
            synthetic = []      # missing symmetric peaks to be added
    
            # For each detected peak, evaluate its n-fold orbit
            for i in range(M):
                y0 = ring_coords[i, 0]
                x0 = ring_coords[i, 1]
    
                present_indices = set()
                missing_positions = []
    
                # Build orbit by rotating this peak n_fold times
                for k in range(n_fold):
                    yk, xk = rotate_point(y0, x0, k)
    
                    best_j = None
                    best_d2 = tol2
    
                    # Find nearest detected peak to (yk, xk)
                    for j in range(M):
                        dy_ = ring_coords[j, 0] - yk
                        dx_ = ring_coords[j, 1] - xk
                        d2 = dy_ * dy_ + dx_ * dx_
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_j = j
    
                    if best_j is not None:
                        present_indices.add(best_j)
                    else:
                        missing_positions.append((yk, xk))
    
                m = len(present_indices)
                completeness = m / float(n_fold)
    
                if completeness < orbit_min_fraction:
                    # Orbit is too incomplete; do not treat it as a real symmetric orbit
                    continue
    
                # This orbit is considered real; act according to sym_mode
                if sym_mode in ("prune", "both", "repair"):
                    for j in present_indices:
                        good_measured.append(ring_coords[j])
    
                if sym_mode in ("repair", "both"):
                    for (yk, xk) in missing_positions:
                        synthetic.append([yk, xk])
    
            # Build ring output according to sym_mode and merge close points so
            # that multiple "repairs" do not create clusters at the same site.
            if sym_mode == "repair":
                # Keep all original peaks plus synthetic peaks from sufficiently
                # complete orbits; merging ensures no over-repair.
                stacks = [ring_coords]
                if synthetic:
                    stacks.append(np.asarray(synthetic, float))
                all_coords = np.vstack(stacks)
                ring_out = _finalize_coords(all_coords, sym_tolerance_px)
    
            elif sym_mode == "prune":
                # Keep only peaks belonging to sufficiently complete orbits
                if good_measured:
                    ring_out = _finalize_coords(np.asarray(good_measured, float),
                                                sym_tolerance_px)
                else:
                    ring_out = np.empty((0, 2), dtype=int)
    
            elif sym_mode == "both":
                # Keep only peaks in sufficiently complete orbits and add synthetic
                # peaks for missing symmetric positions
                stacks = []
                if good_measured:
                    stacks.append(np.asarray(good_measured, float))
                if synthetic:
                    stacks.append(np.asarray(synthetic, float))
                if stacks:
                    all_coords = np.vstack(stacks)
                    ring_out = _finalize_coords(all_coords, sym_tolerance_px)
                else:
                    ring_out = np.empty((0, 2), dtype=int)
    
            else:  # should not reach here due to earlier check
                ring_out = _finalize_coords(ring_coords, sym_tolerance_px)
    
            # Combine center and ring peaks
            if center_out.size > 0:
                if ring_out.size > 0:
                    out = np.vstack([center_out, ring_out])
                else:
                    out = center_out
            else:
                out = ring_out
    
            return out.astype(int)
        
        def _reorder_peaks(coords: np.ndarray,
                           Cy: float,
                           Cx: float,
                           n_fold: int,
                           center_tolerance_px: float,
                           sym_tolerance_px: float) -> np.ndarray:
            """
            Reorder peaks in shells (orbits) of size n_fold.
        
            Strategy
            --------
            - Separate central peaks (within center_tolerance_px) from the rest.
            - On off-center peaks:
                * Work in polar coordinates (radius, angle).
                * For each unused seed peak (starting from smallest radius), build an
                  n_fold rotational orbit by rotating its position by k * (2π / n_fold)
                  around the center and matching the nearest unused peaks within a
                  distance sym_tolerance_px.
                * If a full orbit of size n_fold is found, commit that orbit as one
                  "shell" and mark its peaks as used.
            - After all possible orbits are built:
                * Sort shells by increasing mean radius.
                * Within each shell, sort by increasing angle in [0, 2π).
                * Append any remaining unassigned peaks at the end, sorted by radius.
        
            Coordinates are returned in cartesian form (y, x) with their original
            floating-point values preserved (no rounding to int).
            """
            if coords.size == 0:
                return coords
        
            coords_f = np.asarray(coords, float)
            ys = coords_f[:, 0]
            xs = coords_f[:, 1]
        
            dy = ys - Cy
            dx = xs - Cx
            radii = np.hypot(dx, dy)
        
            # Separate central peaks
            center_mask = radii <= center_tolerance_px
            center_coords = coords_f[center_mask]
            ring_coords = coords_f[~center_mask]
        
            # Optionally sort central peaks by radius (they should all be ~0)
            if center_coords.size > 0:
                cy = center_coords[:, 0]
                cx = center_coords[:, 1]
                cr = np.hypot(cx - Cx, cy - Cy)
                center_order = np.argsort(cr)
                center_coords = center_coords[center_order]
        
            # If no off-center peaks, just return central peaks (floats)
            if ring_coords.size == 0:
                return center_coords
        
            # Polar coordinates for off-center peaks
            ys_r = ring_coords[:, 0]
            xs_r = ring_coords[:, 1]
            dy_r = ys_r - Cy
            dx_r = xs_r - Cx
            radii_r = np.hypot(dx_r, dy_r)
            angles_r = np.arctan2(dy_r, dx_r)
            angles_r = np.mod(angles_r, 2.0 * np.pi)  # map to [0, 2π)
        
            Nr = len(ring_coords)
            used = np.zeros(Nr, dtype=bool)
            tried = np.zeros(Nr, dtype=bool)
        
            # Seed order: increasing radius
            seed_order = np.argsort(radii_r)
        
            theta = 2.0 * np.pi / float(n_fold)
            tol2 = sym_tolerance_px ** 2
        
            def rotate_point(y, x, k):
                """Rotate (y, x) by k * theta about (Cy, Cx)."""
                dy_ = y - Cy
                dx_ = x - Cx
                ang = k * theta
                ca = np.cos(ang)
                sa = np.sin(ang)
                dy_r_ = ca * dy_ - sa * dx_
                dx_r_ = sa * dy_ + ca * dx_
                return Cy + dy_r_, Cx + dx_r_
        
            groups_idx = []
        
            # Build rotational orbits
            for i in seed_order:
                if used[i] or tried[i]:
                    continue
        
                tried[i] = True
                y0 = ring_coords[i, 0]
                x0 = ring_coords[i, 1]
        
                local_indices = []
                local_used = set()
        
                for k in range(n_fold):
                    yk, xk = rotate_point(y0, x0, k)
        
                    best_j = None
                    best_d2 = tol2
        
                    # Find nearest unused peak to the expected rotated position
                    for j in range(Nr):
                        if used[j] or (j in local_used):
                            continue
                        dy_ = ring_coords[j, 0] - yk
                        dx_ = ring_coords[j, 1] - xk
                        d2 = dy_ * dy_ + dx_ * dx_
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_j = j
        
                    if best_j is None:
                        # This seed cannot form a complete n_fold orbit under tolerance
                        local_indices = []
                        break
        
                    local_indices.append(best_j)
                    local_used.add(best_j)
        
                # Commit only full orbits
                if len(local_indices) == n_fold:
                    for j in local_indices:
                        used[j] = True
                    groups_idx.append(local_indices)
        
            # Sort groups (orbits) by mean radius
            group_radii = [np.mean(radii_r[idxs]) for idxs in groups_idx]
            group_order = np.argsort(group_radii)
        
            ordered_ring_list = []
        
            for g in group_order:
                idxs = np.array(groups_idx[g], dtype=int)
                # Sort within group by angle
                ang_g = angles_r[idxs]
                order_g = np.argsort(ang_g)
                ordered_ring_list.append(ring_coords[idxs[order_g]])
        
            if ordered_ring_list:
                ordered_ring = np.vstack(ordered_ring_list)
            else:
                ordered_ring = ring_coords[:0]
        
            # Append any remaining unused peaks (no complete orbit) at the end
            unused_idx = np.where(~used)[0]
            if unused_idx.size > 0:
                # Sort leftover by radius before appending
                order_unused = np.argsort(radii_r[unused_idx])
                remainder = ring_coords[unused_idx[order_unused]]
                if ordered_ring.size > 0:
                    ordered_ring = np.vstack([ordered_ring, remainder])
                else:
                    ordered_ring = remainder
        
            # Combine central and ring peaks (all as float coordinates)
            if center_coords.size > 0:
                combined = np.vstack([center_coords, ordered_ring])
            else:
                combined = ordered_ring
        
            return combined

        
        # -------------------------------------------------------------------------
        # Template-matching peak detection
        # -------------------------------------------------------------------------
    
        dp = self.array  # assumed 2D
    
        # Build template kernel
        r = radius
        w = trench_width
        sz = int(np.ceil(r + w)) * 2 + 1
        cy = cx = (sz - 1) / 2.0
    
        yy, xx = np.indices((sz, sz))
        dist_kernel = np.hypot(xx - cx, yy - cy)
    
        kernel = np.zeros((sz, sz), dtype=float)
        kernel[dist_kernel <= r] = kernel_amp
        trench_mask = (dist_kernel > r) & (dist_kernel <= r + w)
        kernel[trench_mask] = trench_amp
    
        # Cross-correlation
        corr = fftconvolve(dp, kernel[::-1, ::-1], mode='same')
    
        H, W = dp.shape
        Cy, Cx = (H - 1) / 2.0, (W - 1) / 2.0
    
        # Build radial mask (if requested)
        mask = None
        if r_range is not None:
            r_min, r_max = r_range
            yy2, xx2 = np.indices((H, W))
            dist_image = np.hypot(xx2 - Cx, yy2 - Cy)
            mask = (dist_image >= r_min) & (dist_image <= r_max)
    
            # Use only allowed region to define the max for threshold_rel
            valid_corr = corr[mask]
        else:
            valid_corr = corr
    
        # Compute threshold if not provided
        if threshold_abs is None and threshold_rel is not None:
            threshold_abs = threshold_rel * valid_corr.max()
    
        # Strongly suppress correlation outside the annulus to avoid spurious peaks
        if mask is not None:
            if np.issubdtype(corr.dtype, np.floating):
                corr = corr.copy()
            else:
                corr = corr.astype(float, copy=True)
            corr[~mask] = -np.inf  # or corr.min() - 1, but -inf is safest
    
        # Local maxima in correlation map
        coords = peak_local_max(
            corr,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
    
        # Final safety filter: enforce r_range on the returned coords
        if r_range is not None and coords.size > 0:
            y = coords[:, 0]
            x = coords[:, 1]
            dist_coords = np.hypot(x - Cx, y - Cy)
            keep = (dist_coords >= r_min) & (dist_coords <= r_max)
            coords = coords[keep]
    
        # Optional n-fold symmetry enforcement on the full peak array
        if (
            n_fold is not None
            and n_fold >= 2
            and coords.size > 0
            and sym_mode != "none"
        ):
            coords = _enforce_nfold_symmetry(
                coords=coords,
                Cy=Cy,
                Cx=Cx,
                n_fold=n_fold,
                sym_mode=sym_mode,
                sym_tolerance_px=sym_tolerance_px,
                center_tolerance_px=center_tolerance_px,
                orbit_min_fraction=orbit_min_fraction,
            )
        
        # Optional reordering into shells of size n_fold
        if (reorder
            and n_fold is not None
            and n_fold >= 2
            and coords.size > 0
        ):
            coords = _reorder_peaks(
                coords=coords,
                Cy=Cy,
                Cx=Cx,
                n_fold=n_fold,
                center_tolerance_px=center_tolerance_px,
                sym_tolerance_px=sym_tolerance_px
            )
        
        return coords

    #TODO: add docstring
    def clip(self, a_min=1, a_max=None):
        """
        
        """
        
        return ReciprocalSpace(clip_values(self.array, a_min, a_max))
    
    #TODO: add docstring
    def get_bg(self, centers, radius,):
                           
        return ReciprocalSpace(inpaint_diffraction(self.array, centers=centers, radius=radius,))
    
    
    def remove_bg(self, background, bg_frac=1, a_min=1):
    
        """
        Subtracts a fraction of the background from the dataset and clips the 
        result to handle underflows.
        
        Parameters
        ----------
        background : ndarray
            The background data array which must be of the same shape as self.array.
        bg_frac : float, optional
            The fraction of the background to be subtracted from the dataset. 
            Must be between 0 and 1 (inclusive).
        
        Returns
        -------
        ReciprocalSpace
            A new instance of ReciprocalSpace with the background subtracted 
            diffraction pattern.
        
        Raises
        ------
        ValueError
            If bg_frac is not within the required range [0, 1] or 
        """
        if not (0 <= bg_frac <= 1):
            raise ValueError("'bg_frac' must be between 0 and 1, inclusive.")
        
        if background.shape != self.shape:
            raise ValueError("""'background' must match the shape of the diffraction pattern.""")
        
        if type(background) != np.ndarray:
            background = background.array
        
        return ReciprocalSpace(self.array - background*bg_frac).clip(a_min=a_min)


    def get_radialProfile(self,
                          r_min: float = 0,
                          r_max: float = None,
                          plot: bool = False,
                          mask: np.ndarray = None,
                          centers: Union[Sequence[Tuple[float, float]], np.ndarray] = None,
                          r: float = None,
                          title: str = None,
                          return_radialProfile: bool = True,
                          return_integral: bool = False,
                          units: str = None,
                          conv_factor: float = None,
                          **plot_kwargs):
        """
        Compute the mean radial intensity profile (and optionally its integral)
        of this 2D diffraction pattern.

        Parameters
        ----------
        r_min : float
            Minimum radius (in pixels) to include. Defaults to 0.
        r_max : float
            Maximum radius to include. Defaults to distance from center to corner.
        plot : bool
            If True, plot log(Intensity) vs. r (or converted units).
        mask : 2D bool array
            If provided, only True pixels are used.
        centers : array-like of shape (N,2) or (2,N)
            If provided, disks of radius `r` around these (ky,kx) coords are excluded.
        r : float
            Radius of exclusion disks around `centers`. Required if `centers` is not None.
        title : str
            Custom plot title.
        return_radialProfile : bool
            If True, return the 1D radial profile.
        return_integral : bool
            If True, return the integral under the radial profile.
        units : str, optional
            Physical units for the radial axis. Accepts any of:
              - inverse Å⁻¹: "inv_ang", "inv_Ang", "invAng", "invang", "a-1", "A-1"
              - milliradians: "mrad"
              - degrees: "deg", "degree"
            If provided, `conv_factor` must also be given.
        conv_factor : float, optional
            Multiply raw pixel–radii by this to get the chosen `units`.
        **plot_kwargs : dict
            Forwarded to plt.plot(), e.g. color, linestyle, linewidth, marker.

        Returns
        -------
        radial_profile : np.ndarray, shape (M,)
            Mean intensity for radii from r_min up to r_max.
        integral : float
            (Optional) Integral under the radial_profile curve
            (in the same units as the x-axis).
        """

        ky, kx = self.shape
        cy, cx = (ky - 1) / 2.0, (kx - 1) / 2.0

        # default outer radius
        if r_max is None:
            r_max = np.hypot(cy, cx)

        # build distance map
        Y, X = np.indices(self.shape)
        distances = np.hypot(Y - cy, X - cx)

        # initial valid‐pixel mask
        valid = np.ones(self.shape, dtype=bool)
        if mask is not None:
            if mask.shape != self.shape:
                raise ValueError("`mask` must have same shape as data")
            valid &= mask

        # exclude disks around centers if requested
        if centers is not None:
            if r is None:
                raise ValueError("`r` must be provided when `centers` is not None")
            centers_arr = np.asarray(centers)
            # handle shape (2,N)
            if centers_arr.ndim == 2 and centers_arr.shape[0] == 2:
                centers_arr = centers_arr.T
            for (cy_i, cx_i) in centers_arr:
                valid &= (np.hypot(Y - cy_i, X - cx_i) > r)

        # bin into integer radii
        max_bin = int(np.floor(r_max)) + 1
        bin_idx = distances.astype(int)

        # sum and count per bin
        sums   = np.bincount(bin_idx[valid].ravel(),
                             weights=self.array[valid].ravel(),
                             minlength=max_bin)
        counts = np.bincount(bin_idx[valid].ravel(),
                             minlength=max_bin)

        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = sums / counts

        # trim to [r_min, r_max]
        # make radii match the profile length exactly
        radii = np.arange(profile.size)
        sel   = (radii >= r_min) & (radii <= r_max)
        raw_profile = profile[sel]
        raw_radii   = radii[sel]

        # units + conversion
        if units is not None:
            if conv_factor is None:
                raise ValueError("`conv_factor` must be provided when `units` is specified")
            # convert radii to physical units
            phys_r = raw_radii * conv_factor
            # choose axis label
            u = units.lower()
            if u in {'inv_ang','inv_Ang','invang','invAng','a-1','A-1','Å-1'}:
                xlabel = r"Frequency (Å$^{-1}$)"
            elif u == 'mrad':
                xlabel = "Scattering angle (mrad)"
            elif u in {'deg','degree'}:
                xlabel = "Scattering angle (°)"
            else:
                xlabel = f"r ({units})"
        else:
            phys_r = raw_radii
            xlabel = "r (px)"

        if plot:
            plt.figure()
            plt.plot(phys_r, np.log(raw_profile), **plot_kwargs)
            plt.xlabel(xlabel)
            plt.ylabel('log(Intensity)')
            plt.title(title if title else 'Radial Profile')
            plt.xlim(phys_r.min(), phys_r.max())
            plt.show()

        outs = []
        if return_radialProfile:
            outs.append(raw_profile)
        if return_integral:
            integral = np.trapz(raw_profile, phys_r)
            outs.append(integral)

        if not outs:
            return None
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)
    
    def select_peaks(self,
              n_points=None,
              **imshow_kwargs):
        """
        Manually select peaks on this 2D diffraction pattern.
    
        Parameters
        ----------
        n_points : int or None
            Maximum number of points to select with mouse clicks.
            If None, selection continues until the user presses Enter.
        **imshow_kwargs : dict
            Passed directly to ax.imshow(); e.g. vmin=…, vmax=…, cmap='inferno', origin='lower', etc.
    
        Returns
        -------
        coords : np.ndarray, shape (N, 2)
            Array of (y, x) locations (in pixel coords) of the selected points.
        """
    
        fig, ax = plt.subplots()
        # show log of intensity
        img = ax.imshow(np.log(self.array), **imshow_kwargs)
        fig.colorbar(img, ax=ax)
        ax.set_title(
            "Click to select peaks.\n"
            + ("Select up to %d points." % n_points if n_points else "Press Enter when done.")
        )
        # n=n_points if int, else n=0 for unlimited until Enter
        ginput_n = n_points if isinstance(n_points, int) else 0
        pts = plt.ginput(n=ginput_n, timeout=0)
        plt.close(fig)
    
        if len(pts) == 0:
            return np.empty((0, 2), float)
    
        # ginput returns list of (x, y) in data coords; convert to (y, x)
        arr = np.array(pts)
        coords = np.column_stack((arr[:, 1], arr[:, 0]))
        return coords

#%% Real-space Class

# class RealSpace:
    
#     def __init__(self):
#         self.data = np.array(data)

#TODO: make function for visualizing the real-space image along with, 
# optionally, an overlaid cluster map.

#%% Denoising Functions and Classes

#TODO: figure out how to show the parameters and docstring for each of these
class DenoisingMethods:
    
    # =============================================================================
    # Spatial Filters
    # =============================================================================

    def gaussian(self, target_data, kernel_size=3, sigma=1):
        """Apply a Gaussian filter to 2D data.
    
        The Gaussian filter reduces noise by averaging the pixel values within a Gaussian kernel,
        creating a smooth image that minimizes high-frequency noise while preserving edges to some extent.
    
        Parameters
        ----------
        target_data : ndarray
            The 2D data to be denoised.
        kernel_size : tuple of int, optional
        sigma : float, optional
            The standard deviation of the Gaussian distribution (default is 1).
    
        Returns
        -------
        denoised_data : ndarray
            The denoised 2D data.
        
        Notes
        -----
        The Gaussian filter is effective in reducing random noise but may blur edges. 
        Adjust the kernel size and sigma to control the degree of smoothing.
        """
        
        return cv2.GaussianBlur(target_data, (kernel_size,kernel_size), sigma)
    
    
    # Tested successfully
    def median(self, target_data, window_size=5):
        """Apply a median filter to 2D data.
    
        The median filter replaces each pixel value with the median value of its neighborhood,
        effectively removing salt-and-pepper noise while preserving edges.
    
        Parameters
        ----------
        target_data : ndarray
            The 2D data to be denoised.
        window_size : int, optional
            The size of the window (default is 5).
    
        Returns
        -------
        denoised_data : ndarray
            The denoised 2D data.
    
        Notes
        -----
        The median filter is particularly effective for removing salt-and-pepper noise.
        It may be less effective for Gaussian noise.
        """
        return median_filter(target_data, size=(window_size,window_size))


    # Tested successfully on (4/30/2024) for real- and reciprocal-space denoising
    def bilateral(self, target_data, d=9, sigma_color=75, sigma_space=75):
        """Apply a bilateral filter to 2D data.
    
        The bilateral filter smooths the image while maintaining sharp edges by considering both
        the spatial proximity and the intensity difference between pixels.
    
        Parameters
        ----------
        target_data : ndarray
            The 2D data to be denoised.
        d : int, optional
            Diameter of each pixel neighborhood used during filtering (default is 9).
        sigma_color : float, optional
            Filter sigma in the color space (default is 75). 
        sigma_space : float, optional
            Filter sigma in the coordinate space (default is 75). As this parameter gets larger, the
            filter behaves like a regular Gaussian filter.
    
        Returns
        -------
        denoised_data : ndarray
            The denoised 2D data.
        """

        return cv2.bilateralFilter(np.float32(target_data), d, sigma_color, sigma_space)
    
    
    # Tested successfully on (4/30/2024) for real-space denoising
    def non_local_means(self, target_data, h=1.15, patch_size=5, patch_distance=6, fast_mode=True):
        """
        Based on https://scikit-image.org/docs/stable/license.html
        
        Inputs
            patch_size:int, optional
            Size of patches used for denoising
            
            patch_distanceint, optional
            Maximal distance in pixels where to search patches used for denoising
            
            hfloat, optional
            Cut-off distance (in gray levels). The higher h, the more permissive one is in accepting patches. 
            A higher h results in a smoother image, at the expense of blurring features. 
            For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less.
        """
        sigma_est = np.mean(estimate_sigma(target_data, ))
        patch_kw = dict(patch_size=patch_size, patch_distance=patch_distance, )
        
        return denoise_nl_means(target_data, h=h*sigma_est, fast_mode=fast_mode, **patch_kw)


    # Successfully tested on 2D data
    def anisotropic_diffusion(self, target_data, niter=10, kappa=30, gamma=0.2, option=2):
        """
        Generalized Anisotropic Diffusion (Perona-Malik filter) for n-dimensional data.
    
        Parameters:
            target_data (numpy.ndarray): n-dimensional data array.
            niter (int): Number of iterations.
            kappa (float): Conduction coefficient, which controls diffusion.
            gamma (float): Maximum value of .25 for stability, scales the update step.
            option (int): 1 for high contrast edges over low contrast, 2 for wide regions over smaller ones.
    
        Returns:
            numpy.ndarray: Diffused image.
        """
        
        modified_data = np.copy(target_data)
        
        for i in range(niter):
            # Calculate gradients along all axes
            gradients = np.gradient(modified_data)
    
            # Calculate diffusion coefficients
            if option == 1:
                conductance = [np.exp(-(np.abs(g) / kappa) ** 2) for g in gradients]
            elif option == 2:
                conductance = [1 / (1 + (np.abs(g) / kappa) ** 2) for g in gradients]
    
            # Compute flux for each dimension
            flux = [g * c for g, c in zip(gradients, conductance)]
    
            # Initialize divergence array
            divergence = np.zeros_like(modified_data)
    
            # Calculate divergence as the sum of gradients of all flux components
            for f in flux:
                grad_f = np.gradient(f)
                for dim in range(modified_data.ndim):
                    divergence += grad_f[dim]
    
            # Update the image
            modified_data += gamma * divergence
    
        return clip_values(modified_data)

    
    # Tested successfully on (4/30/2024) for real-space denoising
    def total_variation(self, target_data, weight=30, eps=0.0001, max_num_iter=100):
        """
        Based on https://scikit-image.org/docs/stable/license.html

        weight float, optional
        Denoising weight. It is equal to 1/lambda. Therefore, the greater the 
        weight, the more denoising (at the expense of fidelity to image).
        
        eps float, optional
        Tolerance eps > 0 for the stop criterion (compares to absolute value 
        of relative difference of the cost function E): The algorithm stops when
        abs(E_{n-1} - E_n < eps*E_0)
        
        max_num_iter int, optional
        Maximal number of iterations used for the optimization.
        """

        return denoise_tv_chambolle(target_data, weight=weight, eps=eps, max_num_iter=max_num_iter)
    
    # @jit(nopython=True, parallel=True) #new
    def adaptive_median_filter(self, target_data, s=3, sMax=7):
        """
        Apply an adaptive median filter to reduce noise while preserving edges.

        Parameters
        ----------
        target_data : numpy.ndarray
            The single-channel (grayscale) image to denoise.
        s : int, optional
            Initial window size for the median filter.
        sMax : int, optional
            Maximum allowable window size for the median filter.

        Returns
        -------
        filtered_target_data : numpy.ndarray
            The denoised image.

        Notes
        -----
        This filter increases the window size adaptively until a non-noise 
        pixel is found or the maximum window size is reached.
        """
        
        if target_data.ndim != 2:
            raise ValueError("Single channel target_data only")
        
        padded_target_data = np.pad(target_data, sMax//2, mode='constant', constant_values=np.min(target_data))
        H, W = target_data.shape
        filtered_target_data = np.zeros_like(target_data)
        
        # for i in tqdm(range(H), desc = 'Applying adaptive median filter'):
        for i in range(H):
            for j in range(W):
                value = self._process_pixel(padded_target_data, i + sMax//2, j + sMax//2, s, sMax)
                filtered_target_data[i, j] = value

        return filtered_target_data

    # Private method (helper function for adaptive median filter)
    def _process_pixel(self, padded_target_data, y, x, s, sMax):
        """
        Private method (helper function for adaptive median filter): process 
        each pixel by adapting the window size and applying levels A and B checks.

        Parameters
        ----------
        padded_target_data : numpy.ndarray
            The padded input image.
        y : int
            y-coordinate in the padded image.
        x : int
            x-coordinate in the padded image.
        s : int
            Current window size.
        sMax : int
            Maximum window size.

        Returns
        -------
        value : float
            The new pixel value after filtering.

        Notes
        -----
        This method increases the window size until the conditions of Levels A 
        or B are met or the maximum window size is reached.
        """
        
        while True:
            window = padded_target_data[y-s//2:y+s//2+1, x-s//2:x+s//2+1]
            Z_min, Z_med, Z_max = np.min(window), np.median(window), np.max(window)

            if Z_med - Z_min > 0 and Z_med - Z_max < 0:
                return self._level_b(window, Z_min, Z_med, Z_max)
            
            s += 2
            if s > sMax:
                Z_xy = window[window.shape[0]//2, window.shape[1]//2]
                return Z_xy

    # Private method (helper function for adaptive median filter)
    def _level_b(self, window, Z_min, Z_med, Z_max):
        """
        Level B processing to determine the output pixel value based on window statistics.

        Parameters
        ----------
        window : numpy.ndarray
            The current window of pixel values.
        Z_min : float
            Minimum value in the window.
        Z_med : float
            Median value in the window.
        Z_max : float
            Maximum value in the window.

        Returns
        -------
        value : float
            Either the original or the median pixel value, based on conditions.

        Notes
        -----
        This method checks if the central pixel in the window is not an extreme 
        value. If true, it returns the original central pixel value; otherwise, 
        it returns the median value.
        """
        
        Z_xy = window[window.shape[0]//2, window.shape[1]//2]
        
        if Z_xy - Z_min > 0 and Z_xy - Z_max < 0:
            return Z_xy
        else:
            return Z_med
    
    # #
    # def wiener(self, other):
        
        
        
    #     return clip_values(filtered_data)    
    
    # =============================================================================
    # Unsupervised Learning
    # =============================================================================
    
    # #
    # def PCA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    # #
    # def kernelPCA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    # #
    # def fastICA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    def nmf(self, X, n_components=None, init='random', update_H=True, solver='cd', 
            beta_loss='frobenius', tol=0.0001, max_iter=200, alpha_W=0.0, alpha_H='same', 
            l1_ratio=0.0, random_state=None, verbose=0, shuffle=False, return_decomposition=False,
            plot_eigenvalues=False, unfold_domain=None):
        """
        Non-negative Matrix Factorization (NMF) using scikit-learn's `non_negative_factorization`.
        
        Computes a decomposition of matrix X into two non-negative matrices W and H such that:
        X ≈ W @ H
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix to decompose.
        
        n_components : int, optional, default=None
            Number of components to use for the factorization. If not set, all features are kept.
        
        init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, optional, default='random'
            Initialization method to use:
            - 'random': non-negative random matrices
            - 'nndsvd': Nonnegative Double Singular Value Decomposition
            - 'nndsvda': NNDSVD with zeros filled with the average of X
            - 'nndsvdar': NNDSVD with zeros filled with small random values
            - 'custom': Custom matrices W and H must be provided if update_H is True
            
        update_H : bool, default=True
            Whether to update the matrix H in the factorization. If False, H is fixed.
        
        solver : {'cd', 'mu'}, default='cd'
            The numerical solver to use:
            - 'cd': Coordinate Descent
            - 'mu': Multiplicative Update
        
        beta_loss : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}, default='frobenius'
            Beta divergence to be minimized, measuring the distance between X and the dot product WH.
        
        tol : float, default=0.0001
            Tolerance of the stopping condition.
        
        max_iter : int, default=200
            Maximum number of iterations to run the algorithm.
        
        alpha_W : float, default=0.0
            Regularization parameter for W. Set to 0 for no regularization.
        
        alpha_H : float or "same", default="same"
            Regularization parameter for H. Set to 0 for no regularization. If "same", it takes the same value as alpha_W.
        
        l1_ratio : float, default=0.0
            The regularization mixing parameter, with 0 <= l1_ratio <= 1. Determines the balance between L1 and L2 penalties.
        
        random_state : int, RandomState instance or None, default=None
            Used for NMF initialization and in the Coordinate Descent solver. Pass an int for reproducible results.
        
        verbose : int, default=0
            The verbosity level.
        
        shuffle : bool, default=False
            If True, randomize the order of coordinates in the Coordinate Descent solver.
    
        return_decomposition : bool, default=False
            If True, return the decomposition W and H.
    
        plot_scree : bool, default=False
            If True, plot a scree plot showing the variance explained by each component.
        
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            The resulting matrix of the factorization.
        
        H : ndarray of shape (n_components, n_features)
            The resulting matrix of the factorization.
        
        n_iter : int
            The number of iterations run.
    
        reconstruction : ndarray of shape (n_samples, n_features)
            The reconstructed matrix X from W and H, returned if return_reconstruction is True.
        """
        from sklearn.decomposition import non_negative_factorization
        import numpy as np
    
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = X.shape
            X = unfold_tensor(X, unfold_domain)
    
        # Perform NMF
        W, H, n_iter = non_negative_factorization(X, n_components=n_components, init=init,
                                                  update_H=update_H, solver=solver,
                                                  beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                                                  alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                                                  random_state=random_state, verbose=verbose, shuffle=shuffle)
        
        # Calculate the sum of squared elements in each component (row of H)
        component_significance = np.sum(H ** 2, axis=1)
        
        # Sort components by significance
        sorted_indices = np.argsort(-component_significance)
        H = H[sorted_indices]
        W = W[:, sorted_indices]
        component_significance = component_significance[sorted_indices]
        
        # Plotting option for log10 of eigenvalues vs. number of components
        if plot_eigenvalues:
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(component_significance) + 1), np.log10(component_significance), 'o-')
            plt.title('Log10 of Eigenvalues vs. Number of Components')
            plt.xlabel('Component Number')
            plt.ylabel('Log10(Eigenvalue)')
            plt.grid()
            plt.show()
        
        # Reconstruct decomposition
        reconstruction = W @ H
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:               
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            return [reconstruction, [W, H, n_iter]]

        else:
            return reconstruction

    
    # Good results with 'reciprocal' unfolding and rank=50
    def parafac(self, tensor, rank, n_iter_max=100, init='svd', svd='truncated_svd', 
                normalize_factors=False, orthogonalise=False, tol=1e-08, random_state=None, 
                verbose=0, return_errors=False, sparsity=None, l2_reg=0, mask=None, 
                cvg_criterion='abs_rec_error', fixed_modes=None, svd_mask_repeats=5, 
                linesearch=False, callback=None, implementation='tensorly', 
                return_decomposition=False, unfold_domain=None):
        """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
        
        Computes a rank-rank decomposition of tensor such that:
        tensor = [|weights; factors[0], ..., factors[-1] |]
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random', CPTensor}, optional, default is 'svd'
            Type of factor matrix initialization. If a CPTensor is passed, this 
            is directly used for initialization.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape 
            (rank, ), which will contain the norms of the factors.
        
        orthogonalise : bool, optional, default is False
            If True, enforce orthogonality on the factors.
        
        tol : float, optional, default is 1e-08
            Relative reconstruction error tolerance. The algorithm is considered 
            to have found the global minimum when the reconstruction error is less than tol.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        return_errors : bool, optional, default is False
            Activate return of iteration errors.
        
        sparsity : float or int, optional, default is None
            If sparsity is not None, we approximate tensor as a sum of low_rank_component 
            and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). 
            sparsity denotes desired fraction or number of non-zero elements in 
            the sparse_component of the tensor.
        
        l2_reg : float, optional, default is 0
            L2 regularization parameter.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where 
            the values are missing and 1 everywhere else.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', 
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol. 
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified. 
            The last mode cannot be fixed due to error computation.
        
        svd_mask_repeats : int, optional, default is 5
            If using a tensor with masked values, this initializes using SVD 
            multiple times to remove the effect of these missing values on the initialization.
        
        linesearch : bool, optional, default is False
            Whether to perform line search as proposed by Bro.
        
        callback : callable, optional
            Function to call at the end of each iteration.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).
        
        Returns
        -------
        CPTensor(weight, factors) : tuple
            weights : 1D array of shape (rank, )
                All ones if normalize_factors is False (default).
                Weights of the (normalized) factors otherwise.
            factors : list of ndarray
                List of factors of the CP decomposition. Element i is of shape 
                (tensor.shape[i], rank).
            sparse_component : nD array of shape tensor.shape
                Returns only if sparsity is not None.
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithms 
            (if return_errors is True).
        
        Notes
        -----
        CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # CANDECOMP/PARAFAC decomposition via ALS
            result = par(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, normalize_factors=normalize_factors,
                                                    orthogonalise=orthogonalise, tol=tol, random_state=random_state, verbose=verbose, return_errors=return_errors,
                                                    sparsity=sparsity, l2_reg=l2_reg, mask=mask, cvg_criterion=cvg_criterion, fixed_modes=fixed_modes,
                                                    svd_mask_repeats=svd_mask_repeats, linesearch=linesearch, callback=callback)
            if return_errors:
                factors, weights, errors = result
            else:
                factors, weights = result
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            if sparsity is not None:
                sparse_component = result[2] if return_errors else result[1]
                results.append(sparse_component)
            return results
        else:
            return reconstruction

    # Dataset must be 3D
    def parafac2(self, target_data, rank=5, n_iter_max=100, n_iter_parafac=5, 
                 implementation='tensorly', return_decomposition=False, 
                 unfold_domain=None, init='random', svd='truncated_svd', 
                 normalize_factors=False, tol=1e-08, absolute_tol=1e-13, 
                 nn_modes=None, random_state=None, verbose=False, return_errors=False,):
        """
        Apply PARAFAC2 decomposition to input tensor.
        """
        
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        #TODO consider that this will be in all decomposition functions, so probably add this to HyperData...
        if unfold_domain is not None:
        
            original_shape = target_data.shape
        
            target_data = unfold_tensor(target_data, unfold_domain, )
        
        if implementation == 'tensorly':
            
            decomposition = par2(target_data, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, normalize_factors=normalize_factors, 
                 tol=tol, absolute_tol=absolute_tol, nn_modes=nn_modes, random_state=random_state, verbose=verbose, return_errors=return_errors, n_iter_parafac=n_iter_parafac,)
            
            # reconstruction = clip_values(tl.parafac2_tensor.parafac2_to_tensor(decomposition))
            reconstruction = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
            
            
            if unfold_domain is not None:
                
                reconstruction = unfold_tensor(reconstruction, unfold_domain, 
                                               undo=True, original_shape=original_shape, )
            
            if return_decomposition:
                results = []
                results.extend((decomposition, HyperData(reconstruction)))
                return results
                
            else:
                return reconstruction
    
    # Testing...
    def randomised_parafac(self, tensor, rank, n_samples, n_iter_max=100, init='random', 
                           svd='truncated_svd', tol=1e-08, max_stagnation=20, 
                           return_errors=False, random_state=None, verbose=0, 
                           callback=None, implementation='tensorly', 
                           return_decomposition=False, unfold_domain=None):
        """Randomised CP decomposition via sampled ALS
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_samples : int
            Number of samples per ALS step.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'random'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        max_stagnation : int, optional, default is 20
            Maximum allowed number of iterations with no decrease in fit.
        
        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.
        
        return_errors : bool, optional, default is False
            If True, return a list of all errors.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        callback : callable, optional
            Function to call at the end of each iteration.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of shape (tensor.shape[i], rank).
        
        Notes
        -----
        Randomised CP decomposition via sampled ALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Randomised CP decomposition
            factors, weights = rand_parafac(tensor, rank=rank, n_samples=n_samples, 
                                           n_iter_max=n_iter_max, init=init, svd=svd, tol=tol,
                                           max_stagnation=max_stagnation, return_errors=return_errors, 
                                           random_state=random_state, verbose=verbose, callback=callback)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, 
                                           original_shape=original_shape)
        
        if return_decomposition:
            results = [factors, weights, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        else:
            return reconstruction     


    def parafac_power_iteration(self, tensor, rank, n_repeat=10, n_iteration=10, 
                                verbose=0, implementation='tensorly', 
                                return_decomposition=False, unfold_domain=None):
        """CP Decomposition via Robust Tensor Power Iteration
        
        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose.
        
        rank : int
            Rank of the decomposition (number of rank-1 components).
        
        n_repeat : int, optional, default is 10
            Number of initializations to be tried.
        
        n_iteration : int, optional, default is 10
            Number of power iterations.
        
        verbose : bool, optional, default is 0
            Level of verbosity.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).
        
        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.
        
        factors : list of 2-D tl.tensor of shape (size, rank)
            Each column of each factor corresponds to one eigenvector.
        
        Notes
        -----
        CP Decomposition via Robust Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # CP Decomposition via Robust Tensor Power Iteration
            factors, weights = parafac_power_iter(tensor, rank=rank, n_repeat=n_repeat, n_iteration=n_iteration, verbose=verbose)
            
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            return weights, factors, reconstruction
        else:
            return reconstruction


    def symmetric_parafac_power_iteration(self, tensor, rank, n_repeat=10, 
                                          n_iteration=10, verbose=False, 
                                          implementation='tensorly', 
                                          return_decomposition=False, unfold_domain=None):
        """Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration
        
        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose, must be symmetric of shape (size, )*order.
        
        rank : int
            Rank of the decomposition (number of rank-1 components).
        
        n_repeat : int, optional, default is 10
            Number of initializations to be tried.
        
        n_iteration : int, optional, default is 10
            Number of power iterations.
        
        verbose : bool, optional, default is False
            Level of verbosity.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.
        
        factor : 2-D tl.tensor of shape (size, rank)
            Each column corresponds to one eigenvector.
        
        Notes
        -----
        Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration
            factors, weights = sym_parafac_power_iter(tensor, rank=rank, n_repeat=n_repeat,
                                                     n_iteration=n_iteration, verbose=verbose)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            return weights, factor, reconstruction
        else:
            return reconstruction



    def non_negative_parafac_hals(self, tensor, rank, n_iter_max=100, init='svd', 
                    svd='truncated_svd', tol=1e-07, random_state=None, 
                    sparsity_coefficients=None, fixed_modes=None, 
                    nn_modes='all', exact=False, normalize_factors=False, 
                    verbose=False, return_errors=False, cvg_criterion='abs_rec_error', 
                    implementation='tensorly', return_decomposition=False, unfold_domain=None):
        """Non-negative CP decomposition via HALS
        
        Uses Hierarchical ALS (Alternating Least Squares) which updates each factor 
        column-wise (one column at a time while keeping all other columns fixed).
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-07
            Tolerance: the algorithm stops when the variation in the reconstruction 
            error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.
        
        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients on each factor. If None, the algorithm is 
            computed without sparsity.
        
        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.
        
        nn_modes : None, 'all' or array of integers, optional, default is 'all'
            Specify which modes to impose non-negativity constraints on. If 'all', 
            then non-negativity is imposed on all modes.
        
        exact : bool, optional, default is False
            If True, the algorithm gives results with high precision but it needs 
            high computational cost. If False, the algorithm gives an approximate solution.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape 
            (rank, ), which will contain the norms of the factors.
        
        verbose : bool, optional, default is False
            Indicates whether the algorithm prints the successive reconstruction 
            errors or not.
        
        return_errors : bool, optional, default is False
            Indicates whether the algorithm should return all reconstruction 
            errors and computation time of each iteration.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', 
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol. 
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of 
            shape (tensor.shape[i], rank).
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithm 
            (if return_errors is True).
        
        Notes
        -----
        Non-negative CP decomposition via HALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition via HALS
            # TODO modify the error things
            factors, weights = nn_parafac_hals(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, tol=tol,
                                              random_state=random_state, sparsity_coefficients=sparsity_coefficients,
                                              fixed_modes=fixed_modes, nn_modes=nn_modes, exact=exact, normalize_factors=normalize_factors,
                                              verbose=verbose, return_errors=return_errors, cvg_criterion=cvg_criterion)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            results = [factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        else:
            return reconstruction

    def non_negative_parafac(self, tensor, rank, n_iter_max=100, init='svd', 
                             svd='truncated_svd', tol=1e-06, random_state=None,
                             verbose=0, normalize_factors=False, return_errors=False, 
                             mask=None, cvg_criterion='abs_rec_error', fixed_modes=None, 
                             implementation='tensorly', return_decomposition=False, unfold_domain=None):
        """Non-negative CP decomposition using multiplicative updates
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-06
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape (rank, ), which will contain the norms of the factors.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', ALS stops at current iteration if (previous rec_error - current rec_error) < tol. If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified. The last mode cannot be fixed due to error computation.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of shape (tensor.shape[i], rank).
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithm (if return_errors is True).
        
        Notes
        -----
        Non-negative CP decomposition using multiplicative updates.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            original_shape = tensor.shape
            tensor = unfold_tensor(tensor, unfold_domain)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition using multiplicative updates
            factors , weights = nn_parafac(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, tol=tol,
                                         random_state=random_state, verbose=verbose, normalize_factors=normalize_factors,
                                         return_errors=return_errors, mask=mask, cvg_criterion=cvg_criterion, fixed_modes=fixed_modes)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((factors, weights))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
        if return_decomposition:
            results = [factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        else:
            return reconstruction


    def cp_constrained(self, target_data, rank=5, n_iter_max=100, n_iter_max_inner=10, 
                       implementation='tensorly', return_decomposition=False, 
                       unfold_domain=None,):
        """
        Apply constrained PARAFAC decomposition to input tensor.

        Parameters
        ----------
        rank : int
            Number of components for the decomposition.
        return_decomposition : bool, optional
            If True, return the decomposition weights and factors instead of the reconstructed tensor.
        **kwargs
            Additional arguments for the constrained_parafac function.

        Returns
        -------
        results : list
            List of either reconstructed tensors or decomposition results (weights and factors) for each 3D tensor.

        Examples
        --------
        >>> my4Dobject = HyperData(data)
        >>> reconstructed_data = my4Dobject.denoiser.apply_deonising(advanced_method='cp_constrained', unfold_domain='real', 
                                                                     rank=3, n_iter_max=50, tol_outer=1e-6)
        """
        
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        #TODO consider that this will be in all decomposition functions, so probably add this to HyperData...
        if unfold_domain is not None:
        
            original_shape = target_data.shape
        
            target_data = unfold_tensor(target_data, unfold_domain, )
        
        if implementation == 'tensorly':
            
            decomposition = constrained_parafac(target_data, rank=rank, 
                                                n_iter_max=n_iter_max, n_iter_max_inner=n_iter_max_inner,)
            
            reconstruction = tl.cp_to_tensor(decomposition)
            
            if unfold_domain is not None:
                
                reconstruction = unfold_tensor(reconstruction, unfold_domain, 
                                               undo=True, original_shape=original_shape, )
            
            if return_decomposition:
                results = []
                results.extend((decomposition, HyperData(reconstruction)))
                return results
                
            else:
                return reconstruction

          
    
    # def tensor_ring(self, input_tensor, rank, mode=0, svd='truncated_svd', implementation='tensorly', 
    #                 return_decomposition=False, unfold_domain=None, verbose=False, ):
    #     """
    #     Tensor Ring decomposition via recursive SVD
    
    #     Decomposes input_tensor into a sequence of order-3 tensors (factors).
    
    #     Parameters
    #     ----------
    #     input_tensor : tensorly.tensor
    #         The input tensor to decompose.
        
    #     rank : Union[int, List[int]]
    #         Maximum allowable TR rank of the factors. If int, then this is the same for all the factors.
    #         If list, then rank[k] is the rank of the kth factor.
    
    #     mode : int, optional, default is 0
    #         Index of the first factor to compute.
    
    #     svd : str, optional, default is 'truncated_svd'
    #         Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
    
    #     verbose : boolean, optional
    #         Level of verbosity.
    
    #     return_decomposition : boolean, optional
    #         Whether to return the decomposition along with the reconstruction.
    
    #     unfold_domain : any, optional
    #         Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
    
    #     Returns
    #     -------
    #     factors : TR factors
    #         Order-3 tensors of the TR decomposition.
    
    #     Examples
    #     --------
    #     Decompose a tensor into Tensor Ring factors
    #     >>> tr_factors = tensor_ring(tensor, rank=5)
    #     >>> tr_factors.shape
    #     (5, 3, 3)
    
    #     Notes
    #     -----
    #     Tensor Ring decomposition decomposes the input tensor into a sequence of order-3 tensors
    #     (factors) by recursively applying SVD.
    #     """
    #     # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
    #     if unfold_domain is not None:
    #         original_shape = input_tensor.shape
    #         input_tensor = unfold_tensor(input_tensor, unfold_domain)
    
    #     # Tensor Ring decomposition
    #     factors = t_ring(input_tensor, rank=rank, mode=mode, svd=svd, verbose=verbose)
    
    #     # Reconstruct the tensor from TR factors
    #     #TODO: how to get original back?
    #     reconstruction = tl.tt_matrix.tt_matrix_to_tensor(factors)
        
    #     # Re-fold the tensor if it was unfolded
    #     if unfold_domain is not None:
    #         reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
    
    #     if return_decomposition:
    #         results = []
    #         results.extend((factors, HyperData(reconstruction)))
    #         return results
    #     else:
    #         return reconstruction
        
    def tensor_train_matrix(self, tensor, rank='same', svd='truncated_svd', implementation='tensorly',
                            verbose=False, return_decomposition=False, unfold_domain=None):
        """Decompose a tensor into a matrix in tt-format
    
        Decomposes the input tensor into a matrix in Tensor Train (TT) format.
    
        Parameters
        ----------
        tensor : tensorized matrix
    
        rank : 'same', float or int tuple, optional
            If 'same', creates a decomposition with the same number of parameters as tensor.
            If float, creates a decomposition with rank x the number of parameters of tensor.
            Otherwise, the actual rank to be used, e.g., (1, rank_2, ..., 1) of size tensor.ndim//2.
            Note that boundary conditions dictate that the first rank = last rank = 1.
    
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
    
        verbose : boolean, optional
            Level of verbosity.
    
        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.
    
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
    
        Returns
        -------
        reconstruction :
        list containing 'reconstruction' and 'tt_matrix'
    
        Notes
        -----
        Tensor Train (TT) decomposition decomposes the input tensor into a sequence of matrices
        in TT-format by recursively applying SVD.
        """
        
        if implementation == 'tensorly':
    
            # Apply unfolding 
            if unfold_domain is not None:
                original_shape = tensor.shape
                tensor = unfold_tensor(tensor, unfold_domain)
        
            # Tensor Train matrix decomposition
            tt_matrix = tt_mat(tensor, rank=rank, svd=svd, verbose=verbose)
        
            # Reconstruct the matrix from TT decomposition
            reconstruction = tl.tt_matrix_to_tensor(tt_matrix)
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
            if return_decomposition:
                results = []
                results.extend((tt_matrix, HyperData(reconstruction)))
                return results
            else:
                return reconstruction
            

    def tensor_train(self, input_tensor, rank, svd='truncated_svd', verbose=False, 
                     return_decomposition=False, unfold_domain=None,
                     implementation = 'tensorly',):
        """TT decomposition via recursive SVD
    
        Decomposes input_tensor into a sequence of order-3 tensors (factors) – also known as Tensor-Train decomposition.
    
        Parameters
        ----------
        input_tensor : tensorly.tensor
            The input tensor to decompose.
        
        rank : {int, int list}
            Maximum allowable TT rank of the factors. If int, then this is the same for all the factors.
            If int list, then rank[k] is the rank of the kth factor.
    
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
    
        verbose : boolean, optional
            Level of verbosity.
    
        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.
    
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
    
        Returns
        -------
        factors : TT factors
            Order-3 tensors of the TT decomposition.
    
        Notes
        -----
        Tensor-Train (TT) decomposition decomposes the input tensor into a sequence of order-3 tensors
        (factors) by recursively applying SVD.
        """
        
        if implementation == 'tensorly':
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                original_shape = input_tensor.shape
                input_tensor = unfold_tensor(input_tensor, unfold_domain)
        
            # Tensor-Train decomposition
            factors = tt(input_tensor, rank=rank, svd=svd, verbose=verbose)
        
            # Reconstruct the tensor from TT factors
            reconstruction = tl.tt_matrix_to_tensor(factors)
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
        
            if return_decomposition:
                results = []
                results.extend((factors, HyperData(reconstruction)))
                return results
            else:
                return reconstruction
    
    def non_negative_tucker_hals(self, tensor, rank, n_iter_max=100, init='svd', 
                                 svd='truncated_svd', tol=1e-08, sparsity_coefficients=None, 
                                 core_sparsity_coefficient=None, fixed_modes=None, 
                                 random_state=None, verbose=False, normalize_factors=False, 
                                 return_errors=False, exact=False, algorithm='fista', 
                                 return_decomposition=False, unfold_domain=None,
                                 implementation = 'tensorly',):
        """Non-negative Tucker decomposition with HALS
        
        Uses HALS to update each factor column-wise and uses FISTA or active set algorithm to update the core.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients for each factor. If None, the algorithm is computed without sparsity.
        
        core_sparsity_coefficient : array of float, optional, default is None
            Coefficient imposing sparsity on the core when updated with FISTA.
        
        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.
        
        random_state : any, optional
            Random seed or state to initialize the random number generator.
        
        verbose : bool, optional, default is False
            Level of verbosity.
        
        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        exact : bool, optional, default is False
            If True, the HALS NNLS subroutines give results with high precision but with a higher computational cost.
            If False, the algorithm gives an approximate solution.
        
        algorithm : {'fista', 'active_set'}, optional, default is 'fista'
            Non-negative least square solution to update the core.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the Tucker decomposition.
        
        errors : list
            List of reconstruction errors at each iteration of the algorithm (if return_errors is True).
        
        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """
        
        if implementation == 'tensorly':
        
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                original_shape = tensor.shape
                tensor = unfold_tensor(tensor, unfold_domain)
            
            # Non-negative Tucker decomposition with HALS
            factors, core, errors = nnth(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, tol=tol,
                                            sparsity_coefficients=sparsity_coefficients, core_sparsity_coefficient=core_sparsity_coefficient,
                                            fixed_modes=fixed_modes, random_state=random_state, verbose=verbose,
                                            normalize_factors=normalize_factors, return_errors=return_errors, exact=exact,
                                            algorithm=algorithm)
            
            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_to_tensor((core, factors))
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
            
            if return_decomposition:
                results = [factors, core, reconstruction]
                if return_errors:
                    results.append(errors)
                return results
            else:
                return reconstruction
    
    
    def non_negative_tucker(self, tensor, rank, n_iter_max=10, init='svd', tol=0.0001, 
                            random_state=None, verbose=False, return_errors=False, 
                            normalize_factors=False, return_decomposition=False, unfold_domain=None,
                            implementation = 'tensorly',):
        """Non-negative Tucker decomposition
        
        Iterative multiplicative update.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        n_iter_max : int, optional, default is 10
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional
            Level of verbosity.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        core : ndarray
            Positive core of the Tucker decomposition, has shape ranks.
        
        factors : ndarray list
            List of factors of the Tucker decomposition, element i is of shape (tensor.shape[i], rank).
        
        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """
        
        if implementation == 'tensorly':
    
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                original_shape = tensor.shape
                tensor = unfold_tensor(tensor, unfold_domain)
            
            # Non-negative Tucker decomposition
            core, factors, errors = nnt(tensor, rank=rank, n_iter_max=n_iter_max, init=init, tol=tol,
                                                                               random_state=random_state, verbose=verbose, return_errors=return_errors,
                                                                               normalize_factors=normalize_factors)
            
            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_to_tensor((core, factors))
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
            
            if return_decomposition:
                results = [core, factors, reconstruction]
                if return_errors:
                    results.append(errors)
                return results
            else:
                return reconstruction

    def partial_tucker(self, tensor, rank, modes=None, n_iter_max=100, 
                       init='svd', tol=0.0001, svd='truncated_svd', random_state=None, 
                       erbose=False, mask=None, svd_mask_repeats=5, 
                       return_decomposition=False, unfold_domain=None, implementation='tensorly'):
        """Partial Tucker decomposition via Higher Order Orthogonal Iteration (HOI)
        
        Decomposes tensor into a Tucker decomposition exclusively along the provided modes.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        modes : None, int list, optional
            List of the modes on which to perform the decomposition.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, or TuckerTensor, optional, default is 'svd'
            Method to initialize the decomposition. If a TuckerTensor is provided, this is used for initialization.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.tenalg.svd.SVD_FUNS.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional
            Level of verbosity.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).
        
        svd_mask_repeats : int, optional, default is 5
            Number of repetitions for the SVD in case of masking.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.
        
        factors : ndarray list
            List of factors of the Tucker decomposition, with core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes.
        
        Notes
        -----
        Partial Tucker decomposition decomposes the input tensor into a core tensor and factor matrices
        along the specified modes.
        """
        
        if implementation == 'tensorly':
            
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                original_shape = tensor.shape
                tensor = unfold_tensor(tensor, unfold_domain)
            
            # Partial Tucker decomposition
            core, factors = partial_tuck(tensor, rank=rank, modes=modes, n_iter_max=n_iter_max, init=init, tol=tol,
                                                                  svd=svd, random_state=random_state, verbose=verbose, mask=mask,
                                                                  svd_mask_repeats=svd_mask_repeats)
            
            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
            
            if return_decomposition:
                return core, factors, reconstruction
            else:
                return reconstruction

    def tucker(self, tensor, rank, fixed_factors=None, n_iter_max=100, init='svd', 
               return_errors=False, svd='truncated_svd', tol=0.0001, random_state=None, 
               mask=None, verbose=False, return_decomposition=False, unfold_domain=None,
               implementation='tensorly',):
        
        """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)
        
        Decomposes tensor into a Tucker decomposition: tensor = [| core; factors[0], ...factors[-1] |]
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        fixed_factors : int list or None, optional, default is None
            If not None, list of modes for which to keep the factors fixed. Only valid if a Tucker tensor is provided as init.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).
        
        verbose : int, optional
            Level of verbosity.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
            
        implementation : string, optional
            'tensorly' to use Tensorly package implementation or 
            'zhang' to use that in Zhang et al. (2020), see 'Notes' below.
        
        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.
        
        factors : ndarray list
            List of factors of the Tucker decomposition. Its i-th element is of shape (tensor.shape[i], ranks[i]).
        
        Notes
        -----
        Tucker decomposition decomposes the input tensor into a core tensor and factor matrices.
        
        The "zhang" implementation is based on that in Zhang et al. (2020):
            Zhang, C., Han, R., Zhang, A. R., & Voyles, P. M. (2020). 
            "Denoising atomic resolution 4D scanning transmission electron 
            microscopy data with tensor singular value decomposition." 
            Ultramicroscopy, 219, 113123.
        """
        
        if implementation == 'tensorly':
            
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                original_shape = tensor.shape
                tensor = unfold_tensor(tensor, unfold_domain)
            
            # Tucker decomposition
            core, factors, errors = tuck(tensor, rank=rank, fixed_factors=fixed_factors, n_iter_max=n_iter_max, init=init,
                                                                  return_errors=return_errors, svd=svd, tol=tol, random_state=random_state,
                                                                  mask=mask, verbose=verbose)
            
            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = unfold_tensor(reconstruction, unfold_domain, undo=True, original_shape=original_shape)
            
            if return_decomposition:
                results = [core, factors, reconstruction]
                if return_errors:
                    results.append(errors)
                return results
            else:
                return reconstruction

    
    # # =============================================================================
    # # Transform Domain Filtering
    # # =============================================================================
        
    def fourier_filter(self, target_data, mode='pass', r_inner=0, r_outer=None, sigma=10):
        """
        Apply a Fourier filter with Gaussian smoothing to an image using simpler logic.
        
        Parameters:
        image : numpy.ndarray
            The input image to filter.
        mode : str
            'pass' for passing the frequencies within the radius, 'cut' for cutting them out.
        r_inner : int
            The radius for the simple pass/cut or inner radius for band filters.
        r_outer : int, optional
            The outer radius for band filters. If provided, implies a band filter.
        sigma : float
            The standard deviation for the Gaussian used to smooth the filter edges.
        
        Returns:
        numpy.ndarray
            The filtered image.
        """
        
        # Compute the Fourier transform of the image
        f_transform = (fftshift(fft2(target_data)))
        
        rows, cols = target_data.shape
        cy, cx = rows // 2, cols // 2
        x = np.arange(cols) - cx
        y = np.arange(rows) - cy
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
    
        # Initialize the filter mask
        mask = np.zeros_like(target_data, dtype=float)
    
        if r_outer and r_outer > r_inner:
            # Band filter
            band_mask = np.logical_and(R >= r_inner, R <= r_outer)
            if mode == 'pass':
                mask[band_mask] = 1
            elif mode == 'cut':
                mask[band_mask] = 0
                mask = 1 - mask
        else:
            # Simple pass/cut with Gaussian falloff
            if mode == 'pass':
                mask = np.exp(-((R - r_inner) ** 2) / (2 * sigma ** 2))
            elif mode == 'cut':
                mask = 1 - np.exp(-((R - r_inner) ** 2) / (2 * sigma ** 2))
    
        # Smooth the mask edges using a Gaussian filter
        mask = gaussian_filter(mask, sigma=sigma)
    
        # Apply the filter to the Fourier transform
        f_transform_filtered = f_transform * mask
    
        # Inverse Fourier transform to get the filtered image
        filtered_image = np.abs(np.fft.ifft2(f_transform_filtered))

        return filtered_image
        
        # return clip_values(filtered_data) 
    
    # #
    # def wavelet_thresholding(self, other):
    #     """
    #     Fourier filtering.
        
    #     discussed in [reference].
    #     """
        
        
    #     return clip_values(filtered_data) 
    
    # #
    # def curvelet(self, other):
    #     """
        
    #     """
        
    #     return clip_values(filtered_data) 
    
    # # 
    # def block_matching(self, other):
    #     """
    #     Block-matching and 3D/4D Filtering (BM3D or BM4D)
        
    #     Input data must be two- or three-dimensional.
        
    #     Based on [reference]
    #     """
        
    #     return filtered_data
        
#%%

class Denoiser:
    """
    The Denoiser class is designed to apply various denoising techniques to 4D-STEM data. It manages
    different types of denoising methods, including filtering and machine learning methods 
    to improve the quality of the data. The class can handle 2D, 3D, and 4D data, applying the 
    appropriate denoising methods based on the dimensionality of the input data.

    Attributes
    ----------
    target_data : ndarray
        The data to be denoised, which can be 2D, 3D, or 4D.

    methods : DenoisingMethods
        An instance of the DenoisingMethods class, which contains all the available denoising techniques.
        
    Methods
    -------
    denoise(method_name, target_data=None, **kwargs)
        Applies the specified denoising method to the target data using the provided parameters.
    
    apply_denoising(real_space_method=None, reciprocal_space_method=None, 
                    real_space_kwargs=None, reciprocal_space_kwargs=None, **kwargs)
        Applies the specified denoising methods to the target data in real space and/or reciprocal space.
        The method dynamically updates the parameters for the respective methods and handles the data 
        based on its dimensionality. When only one method (real-space or reciprocal-space) is provided,
        direct keyword arguments can be used. When both methods are provided, dictionaries are used 
        to manage the respective parameters.

    Notes
    -----
    - The `denoise` method fetches the appropriate method from the DenoisingMethods instance and applies it to 
      the data. It raises a ValueError if the specified method is not found.
    - The `apply_denoising` method intelligently updates the keyword arguments for real-space and reciprocal-space 
      methods, ensuring that parameters are passed correctly. It supports handling of 2D, 3D, and 4D data, 
      with appropriate processing for each dimensionality.
    - For 2D and 3D data, the method does not support simultaneous application of real-space and reciprocal-space methods 
      and will raise a ValueError if both are provided.
    """
    
    def __init__(self, target_data):
        
        if type(target_data) == np.ndarray:        
            self.array = target_data
            self.ndim = target_data.ndim

        self.methods = DenoisingMethods()
        
        # Make a list of available denoising methods accessible (exclude helper functions starting with "_")
        available_methods = [m for m, v in inspect.getmembers(self.methods, predicate=inspect.ismethod) if not m.startswith('_')]
        self.available_methods = available_methods

    def denoise(self, method_name, target_data=None, **kwargs):
                
        if target_data is None:
            target_data = self.array    

        method = getattr(self.methods, method_name, None)

        if method:
            
            try:
                return method(target_data, **kwargs)
            
            except TypeError as e:
                sig = inspect.signature(method)
                param_names = ', '.join([param.name for param in sig.parameters.values() if param.name != 'target_data'])
                raise TypeError(f"{str(e)}. Valid arguments are: {param_names}")

        else:
            raise ValueError(f"No such method '{method_name}'. Available methods are: {', '.join(self.available_methods)}")
    
    # @jit(nopython=True, parallel=True)
    def apply_denoising(self, real_space_method=None, real_space_kwargs=None, 
                        reciprocal_space_method=None, reciprocal_space_kwargs=None, 
                        advanced_method=None, advanced_kwargs=None, **kwargs):
        
        #TODO: add a way to return possible inputs for each function
        
        if real_space_kwargs is None:
            real_space_kwargs = {}
        if reciprocal_space_kwargs is None:
            reciprocal_space_kwargs = {}
        if advanced_kwargs is None:
            advanced_kwargs = {}
            
        if real_space_method:
            real_space_kwargs.update(kwargs)
        if reciprocal_space_method:
            reciprocal_space_kwargs.update(kwargs)
        if advanced_method:
            advanced_kwargs.update(kwargs)
        
        if real_space_method is None and reciprocal_space_method is None:
                      
            if advanced_method:
                
                #TODO: assert the method is a string and is within the list of methods
                return self.denoise(advanced_method, **advanced_kwargs)
            
            elif advanced_method is not None:
                raise ValueError("The input method is undefined.")
            
            else:
                raise ValueError("No denoising method provided.")
                        
        dims = self.ndim

        # Below, we handle different dimensions accordingly
        if dims == 2:
            if real_space_method and not reciprocal_space_method:
                return self.denoise(real_space_method, **real_space_kwargs)
            elif reciprocal_space_method and not real_space_method:
                return self.denoise(reciprocal_space_method, **reciprocal_space_kwargs)
            elif real_space_method and reciprocal_space_method:
                raise ValueError("Cannot handle both real space and reciprocal space methods simultaneously for 2D data.")

        elif dims == 3:
            if real_space_method and not reciprocal_space_method:
                return np.array([self.denoise(real_space_method, target_data=slice, **real_space_kwargs) for slice in self.array])
            elif reciprocal_space_method and not real_space_method:
                return np.array([self.denoise(reciprocal_space_method, target_data=slice, **reciprocal_space_kwargs) for slice in self.array])
            elif real_space_method and reciprocal_space_method:
                raise ValueError("Cannot handle both real space and reciprocal space methods simultaneously for 3D data.")

        elif dims == 4:
            ry, rx, ky, kx = self.array.shape
            processed_data = np.zeros_like(self.array)

            if real_space_method:
                for i in tqdm(range(ky), desc = 'Filtering real-space images'):
                    for j in range(kx):
                        processed_data[:, :, i, j] = self.denoise(real_space_method, target_data=self.array[:, :, i, j], **real_space_kwargs)

            if reciprocal_space_method:
                for i in tqdm(range(ry), desc = 'Filtering diffraction patterns'):
                    for j in range(rx):
                        processed_data[i, j, :, :] = self.denoise(reciprocal_space_method, target_data=self.array[i, j, :, :], **reciprocal_space_kwargs)

            return HyperData(processed_data)

        else:
            raise ValueError("Unsupported image dimensionality")


            

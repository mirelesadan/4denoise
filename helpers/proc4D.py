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

if __name__ == '__main__':
    import os
    os.chdir("/home/han/Users/adan/pythonCodes/InverseFunction")
    # os.chdir("C:/Users/haloe/Documents/CodeWriting/pythonCodes/HanLab/InverseFunction/")
    import sys
    sys.path.append("/home/han/Users/adan/4denoise")
    # sys.path.append("C:/Users/haloe/Documents/CodeWriting/4Denoise/")


import numpy as np
import h5py

from scipy.stats import mode
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.signal import find_peaks as hist_peaks
import scipy.stats
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy import io
from scipy.linalg import polar
from scipy.fft import fft2, fftshift
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

plt.rcParams['font.family'] = 'Lato'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import time

from skimage.measure import profile_line
from skimage import transform
from skimage import feature
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
import inspect

import numba #new
from numba import jit, prange #new

from kin_diff_sim import generate_library, get_tilts
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
    
#%%

coords = [
    [58, 89],
    [40, 72],
    [46, 47],
    [71, 39],
    [88, 56],
    [82, 81],  # First spots_order ends
    [33, 98],
    [21, 55],
    [52, 21],
    [95, 30],
    [107, 73],
    [76, 108],  # Second spots_order ends
    [51, 116],
    [14, 81],
    [27, 29],
    [77, 12],
    [113, 48],
    [101, 100]  # Third spots_order ends
]

coords_wsData = [(39, 79), (36, 50), (58, 31), (86, 41), (90, 70), (67, 90), (44, 109), (13, 68), (31, 21),
                (82, 13), (114, 52), (95, 101), (72, 120), (17, 98), (7, 39), (54, 2), (110, 22), (119, 81)]

# =============================================================================
# Useful Functions
# =============================================================================

def read_4D(fname, dp_dims = (128, 130), trim_dims = (128,128), trim_meta = True):
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
        dp = read_mat_file(fname)
        
    elif fname_end == 'npy':
        dp = np.load(fname)
        
    else:
        print('This function only supports reading .mat, .raw & .npy files.') 

    # Replace negative and near-zero pixel values with 1
    low_vals_mask = dp < 1
    dp[low_vals_mask] = 1
    
    return dp

def read_mat_file(filename):
    try:
        file = io.loadmat(filename)

        for key in file.keys():
            content = file[key]
            
            if isinstance(content, np.ndarray):
                if len(content.shape) > 1:
                    return content
        else:
            print("No good key not found in the .mat file.")
            return file
        
    except NotImplementedError:
        print("This file may be in HDF5 format. Trying h5py to load.")
        return read_mat_file_h5py(filename)
    
    except Exception as e:
        print(f"Failed to read the .mat file: {e}")
        return None

def read_mat_file_h5py(filename):
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
                print("No good key not found in the .mat file.")
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

def make_mask(centers, r_mask, mask_dim=(128,128), invert=False,):
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
        transformed_arr = (1/4)*array**2 + (1/4)*np.sqrt(3/2)*array**(-1) - (11/8)*array**(-2) + (5/8)*np.sqrt(3/2)*array**(-3) - 1/8
    
    return transformed_arr

def add_poisson_noise(array, counts):
    
    # Remove any negative numbers and normalize
    noisy_array = np.copy(array)
    noisy_array[noisy_array < 0] = 0
    noisy_array /= np.sum(noisy_array)
    
    # Apply Poisson noise
    noisy_array = np.random.default_rng().poisson(noisy_array * counts,) 
    
    return noisy_array

def get_surface_tilt_and_direction(surface, units='rad', show_results=True):
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

    Examples
    --------
    >>> import numpy as np
    >>> surface = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> tilt_magnitude, tilt_direction = calculate_surface_tilt(surface, units='deg', show_results=True)

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
        unit_label = 'Degrees'
    else:
        unit_label = 'Radians'
    
    if show_results:
        plt.figure(figsize=(12, 6))

        # Tilt magnitude
        ax1 = plt.subplot(1, 2, 1)
        cax1 = plt.imshow(tilt_magnitude, cmap='gray')
        plt.title('Tilt Magnitude')
        cb1 = plt.colorbar(cax1, ax=ax1)
        cb1.ax.set_title(unit_label, pad=10)

        # Tilt direction
        ax2 = plt.subplot(1, 2, 2)
        cax2 = plt.imshow(tilt_direction, cmap='hsv')
        plt.title('Tilt Direction')
        cb2 = plt.colorbar(cax2, ax=ax2)
        cb2.ax.set_title(unit_label, pad=10)

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

        # if subplot:
        #     plt.figure(figsize=(12, 6))  # Increase the figure size for better visibility
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(rgb_image)
        #     # plt.title("RGB Image of Field")
        #     plt.axis('off')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(color_wheel, origin='lower')
        #     # plt.title("Color Wheel")
        #     plt.axis('off')
        #     sm = ScalarMappable(cmap=warphase)
        #     sm.set_array([])
        #     # cbar = plt.colorbar(sm, orientation='vertical', ticks=[0, 0.25, 0.5, 0.75, 1])
        #     cbar = plt.colorbar(sm, orientation='vertical', ticks=[0, 0.25, 0.5, 0.75, 1], fraction=0.046, pad=0.04)  # Adjusted size
        #     cbar.set_label('Hue', rotation=270, labelpad=15)
        #     cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
        #     plt.tight_layout()
        #     plt.show()
        # else:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(color_wheel, origin='lower')
        #     plt.axis('off')
        #     plt.show()
            
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

# def calculate_PSNR(noisy_arr, clean_arr):
#     """
    
#     """
    
#     PSNR = 0

def unfold_tensor(tensor, unfold_domain, undo=False, original_shape=None):
    
    #TODO: assert that unfold_domain is a string and is included in a list of possible unfolding methods
    
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
    """
    Function to clip values in array.
    """

    return np.clip(array, a_min, a_max)

from scipy.spatial import ConvexHull
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

# Function specific to simple monolayer data
def get_strains(spots_order, centers_data, ref_centers=None, ang=49, gs=1, plot=True, order=False):
    """
    spots_order = 1,2,3
    """

    if order:
        diff_pos = centers_data
    else:
        diff_pos = centers_data[spots_order-1]

    if ref_centers is None:
        mean_pos = (np.mean(np.mean(diff_pos[40:60, 25:65], axis=0), axis=0)
                    + np.mean(np.mean(diff_pos[40:60, 95:115], axis=0), axis=0)
                    + np.mean(np.mean(diff_pos[40:60, 157:175], axis=0), axis=0)
                    + np.mean(np.mean(diff_pos[40:60, 210:230], axis=0), axis=0))/4
    else:
        mean_pos = ref_centers

    """ g vectors """
    g1 = diff_pos[:, :, gs-1, :] - diff_pos[:, :, gs+2, :]
    g2 = (diff_pos[:, :, gs, :] + diff_pos[:, :, gs+1, :]) / 2 - \
        (diff_pos[:, :, (gs+3) % 6, :] + diff_pos[:, :, (gs+4) % 6, :]) / 2

    print("g1: ", np.mean(g1, axis=(0, 1)))
    print("g2: ", np.mean(g2, axis=(0, 1)))

    """ Reference g's """
    ref_g1 = mean_pos[gs-1] - mean_pos[gs+2]
    ref_g2 = (mean_pos[gs] + mean_pos[gs+1]) / 2 - \
        (mean_pos[(gs+3) % 6] + mean_pos[(gs+4) % 6]) / 2

    """ G array """
    G_ref = np.array([[ref_g2[0], ref_g1[0]],
                      [ref_g2[1], ref_g1[1]]])

    """ R matrices """
    ang = np.radians(ang)
    R1 = np.array([[np.cos(ang), np.sin(ang)],
                  [-np.sin(ang), np.cos(ang)]])

    """ Strain arrays """
    ydim = centers_data.shape[0]
    xdim = centers_data.shape[1]

    exx1 = np.zeros((ydim, xdim))
    eyy1 = np.zeros((ydim, xdim))
    exy1 = np.zeros((ydim, xdim))
    theta1 = np.zeros((ydim, xdim))

    """ Calculations """
    for i in range(ydim):
        for j in range(xdim):
            G1 = np.array([[g2[i][j][0], g1[i][j][0]],
                          [g2[i][j][1], g1[i][j][1]]])
            T = R1@G1@np.linalg.inv(G_ref)@np.linalg.inv(R1)

            R, U = polar(T)

            # Uniaxial
            eyy1[i][j] = 1-U[0, 0]
            exx1[i][j] = 1-U[1, 1]

            # Shear
            exy1[i][j] = U[1, 0]

            # Rotation
            theta1[i][j] = np.arccos(R[0, 1])

    if plot:
        # Plot uncorrected strain
        plotStrain(exx1, 0.06)
        plotStrain(eyy1, 0.03)
        plotStrain(exy1, 0.06)
        plotRot(theta1)

    return exx1, eyy1, exy1, theta1


def plotStrain(strain_data, title='Strain', axis='on', lim_val=0.05):
    """ Strain/Rotation map plotting """

    plt.figure(figsize=(4.5, 10))

    im1 = plt.imshow(strain_data, vmin=-lim_val, vmax=lim_val, cmap='RdBu')

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

def apply_majority_filter(cluster_map, kernel_size=3, iterations=1):
    """
    Apply a majority filter to a 2D cluster map.

    cluster_map: A 2D numpy array with cluster indices.
    kernel_size: The length of the square kernel (must be an odd number).

    Returns a 2D numpy array after applying the majority filter.
    """

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    def majority_filter(pixel_values):
        # Return the mode (most common element) of the array.
        return mode(pixel_values, axis=None)[0][0]

    # Generate a footprint for the filter (square kernel)
    footprint = np.ones((kernel_size, kernel_size))

    # Apply generic filter with the custom majority function
    filtered_map = ndimage.generic_filter(
        cluster_map, majority_filter, footprint=footprint, mode='constant', cval=0)

    # Filter more times if user specified this
    if iterations > 1:
        for i in range(iterations-1):
            filtered_map = ndimage.generic_filter(
                filtered_map, majority_filter, footprint=footprint, mode='constant', cval=0)

    return filtered_map


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

    
def iterative_reconstruction(xGrad, yGrad, y_bds_flat, x_bds_flat, iterations=10, threshold_percent=5, 
                             box_size=(3, 3), plot=True):
    
    """
    Make 3D reconstruction based on xGrad and yGrad information. For each 
    iteration, the gradient sign (without changing its magnitude) is refined 
    for surface continuity.
    """
    xCorr = fix_sign_errors(xGrad, (3, 3))
    yCorr = fix_sign_errors(yGrad, (3, 3))

    for i in range(iterations):
        # Step 1: Obtain a height map "hmap_fixed" using xGrad and yGrad
        
        if plot and i == iterations-1:
            
            plotMap(xCorr, 'X')
            plotMap(yCorr, 'Y')
        
        h_map = reconFromGradDir(yCorr, xCorr, plot=False)
        hmap_fixed = fix_tilt_and_height(h_map, y_bds_flat, x_bds_flat)

        if i < iterations - 1:
            
            # Step 2: Obtain the gradients from the reconstructed map
            xGrad_recon, yGrad_recon = compute_gradients(hmap_fixed)

            # Compare the original, modified gradients xCorr, yCorr with xGrad_recon, yGrad_recon
            xDiff = np.abs(xCorr - xGrad_recon)
            yDiff = np.abs(yCorr - yGrad_recon)

            # Determine threshold based on percentile
            xThreshold = np.percentile(xDiff, 100 - threshold_percent)
            yThreshold = np.percentile(yDiff, 100 - threshold_percent)

            # Flip signs where the difference exceeds the threshold
            xCorr[xDiff > xThreshold] *= -1
            yCorr[yDiff > yThreshold] *= -1
            
            # Step 3: Fix signs of gradients using the same function as before
            xCorr = fix_sign_errors(xCorr, box_size)
            yCorr = fix_sign_errors(yCorr, box_size)
    
    
    if plot: 
        # Finally, plot the resulting height map after all iterations
        plotMap(hmap_fixed, title="Iterated Height Map",
                legendtitle="Height \n (px)")
        
        return hmap_fixed, xCorr, yCorr
        
    else: 
        
        return hmap_fixed

def fix_sign_errors(img, window_shape=(3, 3)):
    
    # Ensure both dimensions of window_shape are odd for center pixel calculation
    if any(dim % 2 == 0 for dim in window_shape):
        raise ValueError("Both dimensions of window_shape must be odd.")

    # Create a copy of the original image to store the corrected values.
    corrected_img = np.copy(img)

    # Calculate the margins
    margin_y = window_shape[0] // 2
    margin_x = window_shape[1] // 2

    # Get the dimensions of the image.
    rows, cols = img.shape

    # Slide through the image using the specified window.
    for i in range(margin_y, rows-margin_y):
        for j in range(margin_x, cols-margin_x):
            
            # Extract the neighborhood.
            window = img[i-margin_y:i+margin_y+1, j-margin_x:j+margin_x+1]

            # Check the sign of the center pixel.
            center_val = img[i, j]

            # If the sign of the sum of the window and the center pixel are different, flip the sign.
            # if np.sign(np.sum(window)) != np.sign(center_val):
            if np.sign(np.median(window)) != np.sign(center_val):
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
    assert len(height_map.shape) == 2,'The height map must be 2-dimensional'
    
    # Initialize the gradient maps with zeros
    xGrad_recon = np.zeros_like(height_map)
    yGrad_recon = np.zeros_like(height_map)

    # Compute x-gradient using center pixel (exclude 1st and last columns)
    xGrad_recon[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / 2
    # Copy the gradient for the edge columns
    xGrad_recon[:, 0] = xGrad_recon[:, 1]
    xGrad_recon[:, -1] = xGrad_recon[:, -2]

    # Compute y-gradient using center pixel (exclude 1st and last rows)
    yGrad_recon[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / 2
    # Copy the gradient for the edge rows
    yGrad_recon[0, :] = yGrad_recon[1, :]
    yGrad_recon[-1, :] = yGrad_recon[-2, :]

    return xGrad_recon, yGrad_recon  

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

def get_gradients(sol_array, rot_angle=0):
    """
    Get directional gradient from solution array (phi, theta). The function 
    assumes the solution array is of shape (rows, cols, 2).
    """

    # Adjust phi by the rotation angle
    phi = sol_array[:, :, 0] - rot_angle*np.pi/180
    theta = sol_array[:, :, 1]

    # Compute gradients
    dzdy = np.tan(theta) * np.cos(phi)
    dzdx = np.tan(theta) * np.sin(phi)

    # Stack gradients
    array_gradients = np.stack((dzdx, dzdy), axis=-1)

    return array_gradients

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
        self.k_shape = (data.shape[2], data.shape[3])
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
    
    
    def alignment(self, r_mask=5, iterations=1, returnStats=False):
        """
        Align the diffraction patterns through the Center of mass of the center beam
        
        Function written by Chuqiao Shi (2022)
        See on GitHub: Chuqiao2333/Hierarchical_Clustering
        
        Modified by Adan J Mireles (April, 2024)
        - Reordered dimensions (x, y) to (y, x)
        - Modified translation from 'ky//2 + com' to '(ky-1)//2 + com'
        """

        y, x, ky, kx = self.shape
        com_y, com_x = self._quickCOM(r_mask=r_mask) 
        cbed_tran = np.copy(self.array)
        std_com = (np.std(com_y), np.std(com_x))
        mean_com = (np.mean(com_y), np.mean(com_x))
        
        print(f'Initial standard deviation statistics (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
        print(f'Initial COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')
        
        for idx in range(iterations):
            print()
            print(f'Processing {y} by {x} real-space positions. Iteration ({idx+1}/{iterations})...')        
            for i in tqdm(range(y), desc = 'Alignment Progress'):
                for j in range(x):
                    afine_tf = transform.AffineTransform(translation=(-(ky-1)//2 + com_y[i,j], -(kx-1)//2 + com_x[i,j]))
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
    
    
    def rotate_dps(self, angle, units='deg'):
        """
        Rotate clockwise all diffraction patterns by angle in degrees
        """
        
        if units == 'rad':
            angle = np.degrees(angle)
        
        if self.ndim == 4:
            y, x, ky, kx = self.shape
            
            new_data = np.zeros((y, x, ky, kx), dtype=self.dtype)
            
            for y_idx in tqdm(range(y), desc = 'Rotating Diffraction Patterns'):
                for x_idx in range(x):
                    new_data[y_idx, x_idx] = rotate(self.array[y_idx, x_idx], angle)
    
    
    def standardize(self):
        """
        Standardize a 4D dataset using NumPy.
                
        Returns
        -------
        standardized_tensor : ndarray
            The standardized 4D tensor with zero mean and unit variance.
        """
        # Calculate the mean and standard deviation along the last dimension
        mean = np.mean(self.array, keepdims=True)
        std = np.std(self.array, keepdims=True)
        
        # Standardize the tensor
        standardized_tensor = (self.array - mean) / std
        
        return HyperData(standardized_tensor)
        
    
    def normalize(self):
        """
        Normalize a 4D dataset to the range [0, 1] using NumPy.
                
        Returns
        -------
        normalized_tensor : ndarray
            The normalized 4D tensor with values in the range [0, 1].
        """
        # Calculate the minimum and maximum values along the last dimension
        min_val = np.min(self.array, keepdims=True)
        max_val = np.max(self.array, keepdims=True)
        
        # Normalize the tensor
        normalized_tensor = (self.array - min_val) / (max_val - min_val)
        
        return HyperData(normalized_tensor)
    
    
    def clip(self, a_min=1, a_max=None):
        """
        
        """
        
        return HyperData(clip_values(self.array, a_min, a_max))
    
        
    def crop(self, ylim=None, xlim=None, kylim=None, kxlim=None):
        """
        Crop a 4D dataset either in the real or the reciprocal domain.
    
        Parameters
        ----------
        ylim : int or tuple, optional
            The vertical real-space limits to be used for cropping. If int, a single row at that index;
            if tuple, defines the start and end indices (inclusive); if None, all rows are included.
        xlim : int or tuple, optional
            The horizontal real-space limits to be used for cropping. If int, a single column at that index;
            if tuple, defines the start and end indices (inclusive); if None, all columns are included.
        kylim : int or tuple, optional
            The vertical reciprocal-space limits to be used for cropping. If int, a single row at that index;
            if tuple, defines the start and end indices (inclusive); if None, all rows are included.
        kxlim : int or tuple, optional
            The horizontal reciprocal-space limits to be used for cropping. If int, a single column at that index;
            if tuple, defines the start and end indices (inclusive); if None, all columns are included.
        """
        
        # Helper function to parse limits
        def parse_limits(limits, max_length):
            if limits is None:
                return slice(None)  # all elements
            elif isinstance(limits, int):
                if limits < 0 or limits >= max_length:
                    raise ValueError("Index out of bounds")
                return slice(limits, limits + 1)  # single element to a slice
            elif isinstance(limits, tuple):
                start, end = limits
                if start < 0 or end >= max_length or start > end:
                    raise ValueError("Invalid slice range or out of bounds")
                return slice(start, end + 1)  # add 1 because the end is exclusive
            else:
                raise ValueError("Limits should be either int or tuple of two ints")
    
        # Parse each set of limits
        ylim_slice = parse_limits(ylim, self.shape[0])
        xlim_slice = parse_limits(xlim, self.shape[1])
        kylim_slice = parse_limits(kylim, self.shape[2])
        kxlim_slice = parse_limits(kxlim, self.shape[3])
    
        return HyperData(self.array[ylim_slice, xlim_slice, kylim_slice, kxlim_slice])
        
    
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
    
    # Must add type of interpolation
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
    
        Example
        --------
        >>> hyperdata_instance = HyperData(data)
        >>> corrected_data_instance = hyperdata_instance.fix_elliptical_distortions(r=20, R=40, interp_method='cubic')
        
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
        for i in range(A):
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
    
    
    def get_dp(self, y=None, x=None, real_mask=None, reciprocal_mask=None):
        """
        Obtain a diffraction pattern from a specific 4D dataset, either at a 
        specific point, averaged over a specified region, or a special diffraction
        such as the dataset's 'mean', 'median', 'max', etc..
        
        Inputs:
            y: (tuple of int, int, or string) If tuple (ymin, ymax), it 
               specifies the vertical coordinate range to average over. If int,
               it specifies a specific vertical coordinate. If string, 'y' must 
               be one of the operations in 'operations' and returns the corresponding
               special diffraction pattern such as the 'mean', 'median', etc.
            x: (tuple of int or int, optional) If tuple (xmin, xmax), it 
               specifies the horizontal coordinate range to average over. If 
               int, it specifies a specific horizontal coordinate.
        
        Returns: 
            dp: (numpy array) Diffraction pattern, either at a specific point, 
            averaged over a region, or some other special diffraction.
        
        Raises:
            ValueError: If inputs are improperly specified or not provided.
        """
        
        if real_mask is not None or reciprocal_mask is not None:
            if real_mask is not None:
                if real_mask.shape == (self.shape[0], self.shape[1]):
                    return ReciprocalSpace(np.mean(self.array[real_mask]), axis=(0))
            else:
                if reciprocal_mask.shape == (self.shape[2], self.shape[3]):
                    return np.mean(self.array[:,:,reciprocal_mask], axis=(2))
        
        operations = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'random': np.random.random,
        }

        if isinstance(y, str):
            if y in operations:
                if y == 'random':
                    y_pos = int(self.shape[0]*operations[y]())
                    x_pos = int(self.shape[1]*operations[y]())
                    print(f"Displaying diffraction pattern at position ({y_pos}, {x_pos})...\n")
                    result = self.array[y_pos, x_pos]
                else:
                    result = operations[y](self.array, axis=(0, 1))
                return ReciprocalSpace(result)
            else:
                valid_operations = ', '.join(f"'{op}'" for op in operations.keys())
                raise ValueError(f"'y' must be an integer, tuple of integers, or string: {valid_operations}.")
       
        elif y is not None or x is not None:
            
            if isinstance(y, tuple) and isinstance(x, tuple):
                ymin, ymax = y
                xmin, xmax = x
                return ReciprocalSpace(np.mean(self.array[ymin:ymax, xmin:xmax, :, :], axis=(0, 1)))
            
            elif isinstance(y, tuple) and isinstance(x, int):
                ymin, ymax = y
                return ReciprocalSpace(np.mean(self.array[ymin:ymax, x, :, :], axis=0))
            
            elif isinstance(y, int) and isinstance(x, tuple):
                xmin, xmax = x
                return ReciprocalSpace(np.mean(self.array[y, xmin:xmax, :, :], axis=0))
            
            elif isinstance(y, int) and isinstance(x, int):
                return ReciprocalSpace(self.array[y, x])
           
            elif isinstance(y, tuple) and x is None:
               ymin, ymax = y
               return ReciprocalSpace(np.mean(self.array[ymin:ymax, :, :, :], axis=(0,1)))
           
            elif isinstance(y, int) and x is None:
               return ReciprocalSpace(np.mean(self.array[y, :, :, :], axis=0))

            elif y is None and isinstance(x, tuple):
               xmin, xmax = x
               return ReciprocalSpace(np.mean(self.array[:, xmin:xmax, :, :], axis=(0,1)))

            elif y is None and isinstance(x, int):
               return ReciprocalSpace(np.mean(self.array[:, x, :, :], axis=0))
            
            else:
                raise ValueError("y and x must be either both tuples, both integers, or one tuple and one integer.")
    
        else:
            raise ValueError("No parameters specified.")
    
    
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
    
    
    def get_virtualImage(self, r_minor, r_major, vmin=None, vmax=None, 
                         grid=True, num_div=10,plotMask=False, gridColor='black',
                         plotAxes=True, returnDetector = False):
        """
        Plot virtual detector image
        - r_minor, r_major specify the dimensions of the annular ring over which to integrate
        - vmin, vmax specify the limiting pixel values to be plotted
        
        returnArrays: if True, returns the virtual image and detector mask
        """
        
        assert self.ndim == 4, "Input dataset must be 4-dimensional"
        
        A,B,C,D = self.shape
        
        # Virtual detector mask
        detector_mask = np.zeros((C,D)) 
        
        # Identify reciprocal space pixels within desired ring-shaped region
        for i in range(C):
            for j in range(D):
                if r_minor<=np.sqrt((i-C/2)**2+(j-D/2)**2)<=r_major:
                    detector_mask[i,j] = 1
        
        masked_image = np.sum(self.apply_mask(r_minor, r_major, a_min=0).array, axis=(2,3))
            
        if vmin:
            vmin = vmin
        else:
            vmin = np.min(masked_image[masked_image>0])
        if vmax:
            vmax = vmax
        else:
            vmax=np.max(masked_image)
            
        plt.imshow(masked_image, vmin=vmin, vmax=vmax, cmap='gray')
        
        if type(num_div) == tuple:
            ydiv, xdiv = num_div
        else:
            ydiv = num_div
            xdiv = num_div
        
        if plotAxes:
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
        
        
    def get_centers(self, r, ref_coords, plotSpots=False, method='CoM'):
        
        assert 2 < self.ndim < 5, "Input dataset must be 3- or 4-dimensional"
        Ny, Nx, _, _ = self.shape
        
        all_centers = np.zeros((Ny, Nx, len(ref_coords), 2))
        
        for i in tqdm(range(Ny), desc="Computing centers of mass"):
            for j in range(Nx):
                
                all_centers[i, j] = self.get_dp(i, j).get_centers(r=r, ref_coords=ref_coords, method='CoM')

        return all_centers


    def get_intensities(self, 
                        r=6,
                        centers=None,  
                        ref_coords=None, 
                        method='CoM',
                        compute_resBg=False,
                        residual_frac=0.9,
                        **resBg_kwargs
                        ):

        """
        Extract Bragg peak intensities from a single DP
        """
        
        assert 2 < self.ndim < 5, "Input dataset must be 3- or 4-dimensional"
        Ny, Nx, _, _ = self.shape
        
        if centers is None:
            centers = self.get_centers(r, ref_coords=ref_coords, method=method)
    
        all_ints = np.zeros((Ny, Nx, centers.shape[-2],))
        
        for i in tqdm(range(Ny), desc="Calculating intensities"):
            for j in range(Nx):
                
                dp = self.get_dp(i, j)
                all_ints[i, j] = dp.get_intensities(r=r, centers=centers[i, j], 
                                                    compute_resBg=compute_resBg, **resBg_kwargs)
                
                # if compute_resBg:
                #     res_bg = dp.get_residualBg(centers=centers[i, j], **resBg_kwargs)
                #     # if isinstance(r, (list, np.ndarray)):
                #     all_ints[i,j] -= res_bg * (np.pi*r**2) * residual_frac
                        
        return all_ints


    # def get_residualBg(self, centers_array, bg_method='grimm_ring', **resBg_kwargs):
    #     """
    #     Compute the residual background for every diffraction pattern in dataset.
    #     """
        
    #     Ny, Nx, n, _ = centers_array.shape
        
    #     if bg_method == 'rings':
    #         res_bgs = np.zeros((Ny, Nx, n))
    #     else:
    #         res_bgs = np.zeros((Ny, Nx))
        
    #     for i in tqdm(range(Ny), desc='Computing residual background values'):
    #         for j in range(Nx):
    #             centers = centers_array[i,j]
    #             res_bgs[i,j] = self.get_dp(i, j).get_residualBg(centers, bg_method=bg_method, **resBg_kwargs)
                
    #     return res_bgs


    def apply_mask(self, r_inner, r_outer=None, a_min=0, mask=None, domain=None):
        """
        Apply a mask to all diffraction patterns in HyperData object.
    
        ...
        
        """
        
        if self.ndim < 2:
            raise ValueError("The data object must be of 2 or greater dimensions.")
        
        ky, kx = self.shape[-2], self.shape[-1]
        
        #TODO: add application of arbitrary mask
        # if mask is not None:
            
        
        if r_outer is not None:
            bool_mask = make_mask(((ky-1)/2, (kx-1)/2), (r_inner, r_outer), mask_dim=(ky, kx),)

        else:            
            bool_mask = make_mask(((ky-1)/2, (kx-1)/2), r_inner, mask_dim=(ky, kx),)
        
        
        return HyperData(self.array * bool_mask).clip(a_min=a_min)
    
    
    def get_clusters(self, n_PCAcomponents, n_clusters, r_centerBeam, std_Threshold = 0.2,
                     plotStdMask=True, plotScree=True, plotClusterMap=True, plot3dClusterMap=False, 
                     filter_size=3, cluster_cmap=None, filter_iterations=1, outer_ring=None, polar=False):
        
        """
        Function that returns cluster dataset 
        """
        
        assert len(self.shape)==4, "'dataset' must be 4-dimensional"
        
        # Remove center beam
        A,B,C,D = self.shape
        
        if not polar:
            dataset_noCenter = self.apply_mask(r_inner=r_centerBeam, r_outer=outer_ring).array
        
        else:
            dataset_noCenter = np.zeros_like(self.array)
            if outer_ring is not None:
                dataset_noCenter[:,:,r_centerBeam:outer_ring] = self.array[:,:,r_centerBeam:outer_ring]
            else:
                dataset_noCenter[:,:,r_centerBeam:] = self.array[:,:,r_centerBeam:]
        
        # Find pixels of high variation
        dataset_stdev = HyperData(dataset_noCenter**0.5).get_stdDev(space='reciprocal').array
            
        # Create a mask where True indicates standard deviation is below the threshold
        low_std_dev_mask = dataset_stdev < std_Threshold*np.max(dataset_stdev)
        
        if plotStdMask:
            plt.imshow(low_std_dev_mask, )
            plt.title(f'High Std. Dev. Mask, std_threshold = {std_Threshold}')
            plt.show()
        
        # Initialiaze high stdev data and modify its values
        dataset_noCenter[:, :, low_std_dev_mask] = 0  
        
        dataset_noCenter = dataset_noCenter.reshape(-1, C*D)
        
        # PCA for dimensionality reduction
        components = n_PCAcomponents
        pca = PCA(n_components=components)
        data_reduced = pca.fit_transform(np.log(clip_values(dataset_noCenter)))
        
        # Scree plot to find the optimal number of components
        variance_ratios = pca.explained_variance_ratio_
        
        if plotScree:
            plt.figure()
            plt.plot(range(1, components+1), variance_ratios, marker='o')
            plt.title("Scree Plot")
            plt.xlabel("Principal Component")
            plt.ylabel("Variance Explained")
            plt.show()
        
        # K-means clustering
        n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0,)
        clusters = kmeans.fit_predict(data_reduced)
        
        if cluster_cmap is not None:
            colormap = plt.cm.get_cmap(cluster_cmap, n_clusters) 
        else:
            colormap = plt.cm.get_cmap('gnuplot', n_clusters,)
        
        if plot3dClusterMap:
            # Plotting 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for i in range(n_clusters):
                # Get the color for the current cluster
                color = colormap(i)
            
                # Scatter plot for each cluster
                ax.scatter(data_reduced[clusters == i, 0],
                           data_reduced[clusters == i, 1],
                           data_reduced[clusters == i, 2],
                           # data_reduced[clusters == i, 3],
                           c=[color], label=f'Cluster {i+1}', s=2, alpha=0.5)
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
    
            plt.show()
        
        # Plotting 2D projection
        # Mapping back to original dimensions
        cluster_map = clusters.reshape(A, B)
        
        if filter_size is not None:
            for i in range(filter_iterations):
                cluster_map = apply_majority_filter(cluster_map,filter_size)
        
        # Creating a color-coded AxB map
        cluster_map_colored = np.zeros((A, B, 3), dtype=np.uint8)
            
        # Iterate over each cluster to color
        for i in range(n_clusters):
            # Convert color to RGB format
            color = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
            # Ensure proper broadcasting by using np.where
            indices = np.where(cluster_map == i)
            cluster_map_colored[indices] = color
        
        plt.figure()
        plt.imshow(cluster_map_colored)
        plt.title(f"Cluster Map ({A}x{B}) with {n_clusters} Clusters")
        plt.colorbar()
        plt.show()
        
        return cluster_map

    def remove_bg(self, background, bg_frac=1,):
        
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
        
        if background.shape != self.array.shape[-2:]:
            raise ValueError("""'background' must match the shape of the last 
                             two dimensions of the diffraction dataset.""")
        
        return HyperData(self.array[:,:] - background*bg_frac).clip()
    
#%%

class ReciprocalSpace:

    def __init__(self, data):
        self.array = data
        self.shape = data.shape
        self.denoiser = Denoiser(data)
    
    def show(self, power=1, title='Diffraction Pattern', logScale=True, 
             axes=True, vmin=None, vmax=None, figsize=(10,10), aspect=None, cmap='turbo'):
        """
        Visualize a desired diffraction pattern
        
        power: (int or float) 
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
        
        if aspect is not None:
            plt.gca().set_aspect(aspect)  
            
        if axes:
            
            ky, kx = self.shape
            
            plt.axis('on')
            plt.xticks(np.arange(0, kx, 5))
            plt.yticks(np.arange(0, ky, 5))
    
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
    
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            ax.set_title(title, fontsize=18)
                          
            plt.title("Intensity", fontsize=14)
    
        else:
    
            plt.axis('off')
    
        plt.show()
    
    #TODO: how to call method from HyperData?
    # def remove_center_beam(self):
    #     """
        
    #     """
    #     return ReciprocalSpace(annular)
    
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
            
        # elif method == 'gradient_ascent':
        #     y_len, x_len = masked_spot_data.shape
        #     com_y, com_x = gradient_ascent(masked_spot_data, ((y_len-1)/2, (x_len-1)/2), learning_rate=0.1, max_iters=100)
            
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
    
    # Updated code
    def get_centers(self, r, ref_coords, plotSpots=False, method='CoM'):
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
                                            r + 1e-10,  method, plotSpots,)
                                       
        return centers

    # Needs update
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
            ints[int_idx] = np.sum(masked_data[round(centers[int_idx][0]-(r+1)):round(centers[int_idx][0]+(r+1)),
                                                round(centers[int_idx][1]-(r+1)):round(centers[int_idx][1]+(r+1))])
        
        if compute_resBg:
            res_bg = self.get_residualBg(centers=centers, **resBg_kwargs)
            ints -= res_bg * (np.pi*r**2) * residual_frac
        
        # if residual_pxBg is not None:
            
        #     # Compute the average number of pixels within masked spots 
        #     if isinstance(r, (list, np.ndarray)):
        #         if type(r) == list:
        #             r = np.array(r)
        #         A = np.pi*np.mean(r)**2
        #     else:
        #         A = np.pi*r**2
            
        #     # Subtract the corresponding mean value of the background from each intensity
        #     residual_int = A*(residual_pxBg)
        #     ints -= residual_frac*residual_int
                
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
        - The 'grimm_ring' method creates a large annular mask that ideally passes 
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
            'rings_mean', and 'grimm_ring'. Default is 'rings'.
        t_ring : float, optional
            Thickness of the ring for the 'grimm_ring' method. If not specified, 
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
            and 'grimm_ring' methods).
        """
        
        bg_methods = ['rings', 'rings_mean', 'grimm_ring']
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
        
        if bg_method=='grimm_ring':
            if isinstance(r_spots, tuple):
                raise ValueError(""""For the 'grimm_ring' method, the input parameter 
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
    
    
    def find_peaks(self,min_sigma=0.85, 
                        max_sigma=2.5, 
                        num_sigma=100, 
                        threshold=85,
                        mask_radial_frac=0.9,
                        distance_factors=(0.8,1.5),
                        search_radius_factor=2.5,
                        radii_multiplier=1.5, 
                        radii_slope=1.38,
                        vmin=4,vmax=14, 
                        return_radii=False,
                        axes=False, plotCenters=True,
                        power=1, order_length=None,
                        method='CoM',
                   ):
        
        A, B = self.shape
        
        # Define region where spots are to be found
        if type(mask_radial_frac) == float:
            mask = make_mask(((A-1)//2,(B-1)//2), r_mask=(A+B)//4*mask_radial_frac, mask_dim=self.shape) 
        elif type(mask_radial_frac) == tuple:
            mask = make_mask(((A-1)//2,(B-1)//2), 
                             r_mask=((A+B)//4*mask_radial_frac[0], (A+B)//4*mask_radial_frac[1]),
                             mask_dim=self.shape)
            mask_radial_frac = mask_radial_frac[1]
            
        # Extract centers and radii automatically
        blobs_log = feature.blob_log(self.array**power*mask, min_sigma=min_sigma, 
                                     max_sigma=max_sigma, 
                                     num_sigma=num_sigma, 
                                     threshold=threshold)
        # Unpack values
        r_vals = blobs_log[:,2]
        blobs_array = np.array(blobs_log[:, :2])
    
        # Calculate pairwise distances between blobs. The result is a square matrix.
        distances = cdist(blobs_array, blobs_array)
        # Set diagonal to np.inf to ignore self-comparison.
        np.fill_diagonal(distances, np.inf)
        # Find the distance to the nearest neighbor for each blob.
        nearest_neighbor_dist = distances.min(axis=1)
    
        # Define your distance thresholds.
        
        min_distance_factor, max_distance_factor = distance_factors
        min_threshold = np.mean(np.min(distances,axis=0))*min_distance_factor  # Minimum acceptable distance to the nearest neighbor.
        max_threshold = np.mean(np.min(distances,axis=0))*max_distance_factor  # Maximum acceptable distance to the nearest neighbor.
    
        # Filter blobs. Keep a blob if its nearest neighbor is neither too close nor too far.
        filtered_indices = np.where((nearest_neighbor_dist >= min_threshold) 
                                    & (nearest_neighbor_dist <= max_threshold)
                                    & (((blobs_array[:,0]-A//2)**2 + (blobs_array[:,1]-B//2)**2)**.5 <= (A+B)//4*mask_radial_frac*(8/9)))[0]
    
        # Apply filter
        blobs_log = np.zeros((len(filtered_indices), 3))
        blobs_log[:, :2] = blobs_array[filtered_indices]
        blobs_log[:, 2] = r_vals[filtered_indices]

        
        # Refine peak centers using CoM
        peak_centers = self.get_centers(r=blobs_log[:,2]*search_radius_factor,
                                       ref_coords=blobs_log[:,:2], method=method)
        
        # Sort peaks by their distance to center beam and, optionally, by their 
        # angular position for each Bragg peak order
        peak_centers = sort_peaks(peak_centers, 
                                    ((A-1)//2, (B-1)//2), 
                                    order_length=order_length)
        
        distance_to_center = np.linalg.norm(peak_centers-(A+B)//4,axis=1)
        radii = blobs_log[:,2]*radii_multiplier + radii_slope*(1 - distance_to_center/np.max(distance_to_center))
        
        blobs_log[:, :2] = peak_centers
        blobs_log[:, 2] = radii
        
        if plotCenters:
        
            plt.figure(figsize=(10, 10))
            plt.imshow(np.log(self.array), cmap='turbo',vmin=vmin,vmax=vmax)
            
            if not axes:
                plt.axis('off')
        
            # Circle around each blob
            for blob in blobs_log:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                plt.gca().add_patch(c)
        
            plt.show()
        
        if return_radii:            
            return peak_centers, radii
        else:
            return peak_centers
    
    
    def clip(self, a_min=1, a_max=None):
        """
        
        """
        
        return ReciprocalSpace(clip_values(self.array, a_min, a_max))
    
    
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

#%% Denoising Functions and Classes

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


            
# %% Load 4D data

if __name__ == '__main__':

    # Load data from respective directories
    
    # PC file location
    file_path = 'C:/Users/haloe/Documents/CodeWriting/pythonCodes/HanLab/Ripple/scan_x256_y256.raw'
    
    """Read the 4D dataset as a numpy array from .mat file.
       the shape of the data is (x, y, kx, ky)."""
    data = read_4D(file_path)
    
    # Change the order in real space
    # In python, the default (0,0) is on the top left, however, in EMPAD, it acquires data from the bottom right.
    data = np.flip(data, axis=2)
    data = alignment(data)[0]
    
    np.save('rawRippleData.npy', data)
    
    # Load raw data (aligned)   
    rippleData = np.load('rawRippleData.npy')[92:184, 65:95]
    """
    Approximate center of first thru third order spots
    
    The coordinates were estimated based on approximate location of spots observed
    within average diffraction pattern. 
    """
    
#%%

    # Testing center beam removal
    
    ones = np.ones((7,7))
    structured = HyperData(np.array([[ones, ones*2],
                                     [ones*3, ones*4]]))
    original_shape = structured.shape
    print(structured.array)
   
    # From (0, 1, 2, 3, ...) we swap 1 and 2
    tensor = np.transpose(structured.array, (2, 0, 3, 1 ))
    
    new_tensor_shape = (original_shape[2]*original_shape[0],) + (original_shape[3]*original_shape[1],)
    tensor = tensor.reshape(new_tensor_shape) 

    
    intermediate_shape = (original_shape[2], original_shape[0], original_shape[3], original_shape[1],)
    tensor = tensor.reshape(intermediate_shape)
    # print(tensor)
    tensor = np.transpose(tensor, (1, 3, 0, 2))
    
    
    print(structured.apply_mask(1., 2.5, 888).array)
    
    mask1 = make_mask((50//2,50//2), 10, mask_dim=(50,50), invert=True)
    mask2 = make_mask((50//2,50//2), 20, mask_dim=(50,50), invert=False)
    
    plt.imshow(mask1)
    plt.show()
    
    plt.imshow(mask2)
    plt.show()
    
    plt.imshow(np.logical_and(mask1, mask2))
    plt.show()
    
    unfolded_structured_Real = unfold_tensor(structured.array,'coordinate_aligned_real')
    structured = unfold_tensor(unfolded_structured_Real,'coordinate_aligned_real', undo=True, original_shape=original_shape)
    
    unfolded_structured_Reciprocal = unfold_tensor(structured.array,'coordinate_aligned_reciprocal')
    structured = unfold_tensor(unfolded_structured_Reciprocal,'coordinate_aligned_reciprocal', undo=True, original_shape=original_shape)

    
#%% Ripple Data Testing
    
    rippleData = np.load('rawRippleData.npy')[92:184, 65:95]
    rippleData = HyperData(rippleData)
    rippleData = rippleData.alignment(7)[0]
    rippleData.plotVirtualImage(30, 55, grid=False, plotAxes=False)
    # rippleData.get_dp(special='random').show(vmin=4)
    

    corrected_rippleData = rippleData.fix_elliptical_distortions(r=42,R=50, interp_method='linear', 
                                                                 return_fix=True, show_ellipse=True)
    
#%%
    
    transformed = unfold_tensor(rippleData, 'coordinate_aligned_reciprocal')
    
    resized = cv2.resize((transformed), (20000,20000))

    plt.imshow(np.log(transformed), vmin=4, cmap='turbo',)
    plt.show()
    
    plt.imshow(np.log(resized), vmin=4, cmap='turbo',)
    plt.show()
    
    fourier_transformed = (fftshift(fft2(transformed)))#
    fourier_transformed = np.log(np.abs(fourier_transformed))
    
    resized_ft = cv2.resize((fourier_transformed), (30,92))

    
    plt.imshow(fftshift(fourier_transformed), cmap='gray_r',vmin=12, vmax=19)
    plt.axis('off')
    plt.show()
    
    filtered = median_filter(fourier_transformed, 3)
    transformed_back = np.abs(np.fft.ifft2(filtered))
    
    fourier_folded_back = unfold_tensor((fourier_transformed), 'coordinate_aligned_real',
                                undo=True, original_shape=rippleData.shape)
    
    fourier_folded_back = HyperData(np.abs(fourier_folded_back))
    
    # Denoiser(transformed).denoise('fourier_filter', mode='pass', r_inner=0, r_outer=None, sigma=10)

    #%%

    denoised_ripple = corrected_rippleData.denoiser.apply_denoising(advanced_method='cp_constrained', 
                                                                    rank=200, unfold_domain='real', n_iter_max=100, )
    # denoised_ripple = corrected_rippleData.denoiser.apply_denoising(advanced_method='cp_constrained', rank=400,
    #                                                                 unfold_domain='coordinate_aligned_real', n_iter_max=1000, )
    # rank=5, n_iter_max=100, n_iter_max_inner=10,
    denoised_ripple = HyperData(denoised_ripple)
    corrected_rippleData.plotVirtualImage(45, 55, grid=False)
    denoised_ripple.plotVirtualImage(45, 55, grid=False)
    
    idx1, idx2 = int(np.random.random()*corrected_rippleData.shape[0]), int(np.random.random()*corrected_rippleData.shape[1])
    corrected_rippleData.get_dp(idx1, idx2).show(vmin=4)
    denoised_ripple.get_dp(idx1, idx2).show(vmin=4)
    
    #%%
    # corrected_rippleData.fix_elliptical_distortions(r=42,R=50, interp_method='linear', 
    #                                                              return_fix=False, show_ellipse=True,)
    
    # rippleData.get_dp(special='mean').show(vmin=4,vmax=14,title="Mean Diffraction Before Elliptical Distortion Correction")
    # corrected_rippleData.get_dp(special='mean').show(vmin=4,vmax=14,title="Mean Diffraction After Elliptical Distortion Correction")
    # plt.imshow(rippleData.get_dp(special='mean').array - corrected_rippleData.get_dp(special='mean').array,cmap='RdBu',vmin=-1000,vmax=1000)
       
    # peak_centers, radii, dp_bg = corrected_rippleData.get_dp(special='mean').remove_bg(min_sigma=1, 
    #                                                                         max_sigma=2, 
    #                                                                         num_sigma=50, 
    #                                                                         threshold=50,
    #                                                                         bg_frac=0.99,
    #                                                                         mask_radial_frac=1.2,
    #                                                                         min_distance_factor=0.25,
    #                                                                         max_distance_factor=10,
    #                                                                         radii_multiplier=3,)
    # ReciprocalSpace(dp_bg).show(vmin=4)
    
    rippleData_nobg = HyperData(clip_values(corrected_rippleData.array - 0.8*dp_bg))
    
    start = time.perf_counter()
    rippleData_denoised = rippleData_nobg.denoiser.apply_denoising(real_space_method='adaptive_median_filter',
                                                      real_space_kwargs={'s':3, 'sMax': 7})
    end = time.perf_counter()
    print("It took ", end-start, " seconds." )
    
    # np.save('singleRipple_denoised_noBg.npy', rippleData_denoised.array)
    
    sample = rippleData.get_dp(special='random')
    sample.show(logScale=False, power=0.2, vmin=2.5)
    
    
    denoised_sample = ReciprocalSpace(DenoisingMethods.fourier_filter(1,target_data=sample.array, 
                                                                        r_inner=1.3, r_outer=75, mode='pass',sigma=2))
    denoised_sample.show(logScale=False, power=0.2, vmin=2.5)

    transformed = (fftshift(fft2(sample.array)))
    plt.imshow(np.abs(transformed**0.5))
    other_sample = ReciprocalSpace(np.abs((np.fft.ifft2(transformed))))
    other_sample.show(logScale=False, power=0.2, vmin=2.5)

    
    # HyperData_denoised = corrected_rippleData.denoiser.apply_denoising(real_space_method='anisotropic_diffusion',
    #                                                    real_space_kwargs={'niter':10, 'kappa': 30, 'gamma': 0.25, 'option': 2})
                                                      # niter=10, kappa=30, gamma=0.25, option=2)
    
    cropped_ripple = HyperData(rippleData_nobg.array[2:20, 2:-1])
    cropped_ripple.plotVirtualImage(45,55, grid=False)
    
    cropped_ripple_denoised = cropped_ripple.denoiser.apply_denoising(real_space_method='adaptive_median_filter',
                                                                      real_space_kwargs={'s':3, 'sMax': 5})
    cropped_ripple_denoised.plotVirtualImage(45,55, grid=False)
    
    idx1, idx2 = int(np.random.random()*cropped_ripple.shape[0]), int(np.random.random()*cropped_ripple.shape[1])
    
    cropped_ripple.get_dp(idx1,idx2).show(vmin=4)
    cropped_ripple_denoised.get_dp(idx1,idx2).show(vmin=4)

    # Denoise corrected ripple
    corrected_rippleData_denoised = corrected_rippleData.denoiser.apply_denoising('cp')
    idx1, idx2 = int(np.random.random()*corrected_rippleData.shape[0]), int(np.random.random()*corrected_rippleData.shape[1])
    corrected_rippleData.get_dp(idx1,idx2).show(vmin=4,vmax=14)
    corrected_rippleData_denoised = HyperData(clip_values(corrected_rippleData_denoised.array))
    corrected_rippleData_denoised.get_dp(idx1,idx2).show(logScale=True,vmin=4,vmax=14)
    corrected_rippleData.plotVirtualImage(45, 55, grid=False)
    corrected_rippleData_denoised.plotVirtualImage(45, 55, grid=False)
    
    
    
    #%%
    
    im, _ = rippleData.plotVirtualImage(20,40,plotAxes=False,returnArrays=True,)
    # im2, _ = HyperData(rippleData_denoised.array[1:-1,1:-1]).plotVirtualImage(40,60,plotAxes=False,vmin=vmin, vmax=vmax, returnArrays=True)
    im2, _ = rippleData_denoised.plotVirtualImage(20,40,plotAxes=False,returnArrays=True)
    
    rippleData.get_dp(60,18).show(vmin=4)
    rippleData_denoised.get_dp(60,18).show(vmin=4)
    
  
    
    #%% Anisotropic diffusion testing
       
    singleRipple = np.load('singleRipple.npy')
    HyperData = HyperData(clip_values(singleRipple))
    denoised_real = HyperData.denoiser.apply_denoising(real_space_method='anisotropic_diffusion',
                                                      real_space_kwargs={'niter':15, 'kappa': 25, 'gamma': 0.3, 'option': 2})
    
    denoised_reciprocal = HyperData.denoiser.apply_denoising(reciprocal_space_method='anisotropic_diffusion',
                                                      reciprocal_space_kwargs={'niter':15, 'kappa': 25, 'gamma': 0.3, 'option': 2})
    
    data2denoise = Denoiser(HyperData.array)
    denoised_data = data2denoise.denoise(method_name='anisotropic_diffusion', niter=15, kappa=25, gamma=0.3, option=2)
    denoised_full4D = HyperData(denoised_data)
    
    HyperData.get_dp(50,20).show(vmin=4, axes=False, )
    denoised_real.get_dp(50,20).show(vmin=4, axes=False,)
    denoised_reciprocal.get_dp(50,20).show(vmin=4, axes=False,)
    denoised_full4D.get_dp(50,20).show(vmin=4, axes=False,)
    HyperData.plotVirtualImage(40,60,plotAxes=False)
    denoised_real.plotVirtualImage(40,60,plotAxes=False)
    denoised_reciprocal.plotVirtualImage(40,60,plotAxes=False)
    denoised_full4D.plotVirtualImage(40,60,plotAxes=False)   
    
    #%%    
    
    def plot_regression_for_both_axes(height_map):
        
        """
        Calculate and plot linear regression along 'y' and 'x' axes in 
        reconstructed surface
        """
        
        def plot_regression(ave_height, axis_label):
            x = np.arange(ave_height.size)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, ave_height, rcond=None)[0]
    
            # Plotting
            plt.scatter(x, ave_height, label='ave_height', s=5)
            plt.plot(x, slope*x + intercept, 'r', label='Fitting Line')
            plt.legend()
            plt.xlabel(axis_label + ' Index')
            plt.ylabel('Average Height (nm)')
            plt.title(
                f'Linear Regression on {axis_label} with Slope = {slope:.5f}')
            plt.grid(True)
            plt.show()
    
        # 1. Linear regression and plotting for x-axis
        ave_height_x = np.mean(height_map, axis=0)
        plot_regression(ave_height_x, 'X')
    
        # 2. Linear regression and plotting for y-axis
        ave_height_y = np.mean(height_map, axis=1)
        plot_regression(ave_height_y, 'Y')
    

    
    
    def plotAllHeights(height_map):
        """
        Plot all heights with different color for each row
        """
        # Create an array of colors for each row of the height map
        colors = plt.cm.jet(np.linspace(0, 1, height_map[20:84].shape[0]))
    
        # Define desired figure size (width, height)
        plt.figure(figsize=(10, 2))
    
        # Plot each row in a different color
        for i in range(height_map[20:84].shape[0]):
            plt.scatter(np.arange(height_map[20:84].shape[1]),
                        height_map[20:84][i, :], color=colors[i], s=2, label=f'y={i}')
        # for i in range(exx[20:84].shape[0]):
        #     plt.scatter(np.arange(exx[20:84].shape[1]), exx[20:84][i, :], color=colors[i], s=2, label=f'y={i}')
    
        plt.title('Side View of Heights')
        plt.xlabel('X-axis (size 256)')
        plt.ylabel('Height (nm)')
        plt.xlim(0, 256)  # Setting x-limit
        plt.grid(True)
        # Uncomment the next line if you want to display the legend
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
        plt.show()
    
    
    def plotAveHeightProfile(height_map):
        average_profile = np.mean(height_map[20:84], axis=0)
        x = np.linspace(0, 255, 256)
        fig, ax = plt.subplots(figsize=(11, 5))  # (width, height) in inches
    
        # ax.scatter(x, average_profile, label='ave_height', s=5, c='black')
        ax.plot(x, average_profile, color='black', label='ave_height', linewidth=3)
    
        ax.set_xlabel('X-Index', fontsize=20)
        ax.set_ylabel('Average Height (nm)', fontsize=20)
        ax.set_title('Average Height Profile')
        ax.grid(True)
        ax.set_xlim(0, 256)  # Setting x-limit
        ax.tick_params(axis='both', labelsize=18)  # Adjust 14 to your preference
    
        plt.show()
    
    
    def plotStrain(strain_data, title='Strain',axis='on',lim_val=0.05):
        """ Strain/Rotation map plotting """
    
        plt.figure(figsize=(4.5, 10))
        
        im1 = plt.imshow(strain_data, vmin=-lim_val, vmax=lim_val, cmap='RdBu')
    
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
    
    
    def plotRot(rot_data):
        # Plot the ADF images
        theta_degree = rot_data / np.pi * 180
    
        plt.figure(figsize=(10, 10))
    
        im1 = plt.imshow(-np.reshape(theta_degree-90, (110, 256)),
                         vmin=-2, vmax=2, cmap='RdBu')
        plt.xticks([]), plt.yticks([])
        plt.title('Rotation (deg)')
        # plt.yticks(np.arange(0, 60, step=10), size = 20)
        # plt.grid()
        # plt.xlabel('Theta',size = 30)
        # plt.ylabel('R',size = 30)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    
        plt.colorbar(im1, cax=cax)
        cb = plt.colorbar(im1, cax=cax)
        cb.ax.tick_params(labelsize=10)
        plt.show()    
    
    """Other functions"""
    
    def compute_path_length(row_data):
        """ Compute the path length for a single row """
        return np.sum(np.sqrt(1 + np.diff(row_data)**2))
    
    def compute_radial_profile(img):
        # 1. Check the Input Image Shape
        assert img.shape == (128, 128), f"Unexpected image shape: {img.shape}"
    
        # Calculate the distance of each pixel from the center
        y, x = np.indices((img.shape))
        center = np.array([64, 64])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
    
        # Create mask for valid values
        mask = img > 0.0001
    
        tbin = np.bincount(r[mask].ravel(), img[mask].ravel())
        nr = np.bincount(r[mask].ravel())
    
        radialprofile = tbin / nr
        return radialprofile
    
    
    def plotRadProf(data, alpha=0.1, title='Overlay of radial profiles', bottom=0, top=10, left=0, right=90):
        
        # Calculate and plot radial profile for each 128x128 image
        plt.figure(figsize=(10, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                img = data[i, j]
                radialprofile = compute_radial_profile(img)
                # Use alpha to manage overlay opacity
                plt.plot(np.log(radialprofile), color='blue', alpha=alpha)
        plt.ylim(bottom=bottom, top=top)
        plt.xlim(left=left, right=right)
        plt.xlabel("Distance from center")
        plt.ylabel("log(Intensity)")
        plt.title(title)
        plt.show()
    
    
    def compute_radial_distances(img):
        # Calculate the distance of each pixel from the center
        y, x = np.indices((img.shape))
        center = np.array([64, 64])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return r, img.ravel()
    
    
    def plotRadDist(data, alpha=0.1, title="Radial distances vs Intensities", bottom=0, top=10, left=0, right=90):
    
        # Calculate and plot radial distances vs intensities for each 128x128 image
        plt.figure(figsize=(10, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                img = data[i, j]
                r, intensities = compute_radial_distances(img)
                # Using small points with some transparency
                plt.scatter(r.ravel(), np.log(intensities),
                            color='blue', s=1, alpha=alpha)
        plt.ylim(bottom=bottom, top=top)
        plt.xlim(left=left, right=right)
        plt.xlabel("Distance from center")
        plt.ylabel("Intensity")
        plt.title(title)
        plt.show()
    
    
    def plotMap(array, angle=None, title=None, legendtitle=None):
        """
        angle = 'phi' or 'theta' or 'X' or 'Y' 
        """
    
        plt.figure(figsize=(10, 10))
        
        lim = np.max(np.abs(array))
        
        if (angle == 'X' or angle == 'Y'):
    
            im1 = plt.imshow(array, cmap='RdBu',vmin=-lim,vmax=lim)
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_title(rf"Directional {angle}-Gradient", fontsize=14)
            plt.title("Grad. \n Val.", fontsize=10)
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            plt.show()
    
        elif (angle == 'phi' or angle == 'theta'):
    
            if angle == 'phi':
                im1 = plt.imshow(array, cmap='hsv', vmin=0, vmax=360)
            else:
                im1 = plt.imshow(array, cmap='binary_r')
    
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
    
            if angle == 'phi':
                ax.set_title(r"Tilt Axis Rotation Map $(\phi)$", fontsize=14)
                plt.title("Rotation \n (deg)", fontsize=10)
                cb = plt.colorbar(im1, cax=cax, ticks=np.arange(0, 420, 60))
            else:
                ax.set_title(r"Surface Tilt Map $(\theta)$", fontsize=14)
                plt.title("Tilt \n (deg)", fontsize=10)
                cb = plt.colorbar(im1, cax=cax,)
    
            cb.ax.tick_params(labelsize=15)
            plt.show()
    
        else:
    
            im1 = plt.imshow(array, cmap='turbo')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_title(f"{title}", fontsize=14)
            plt.title(f"{legendtitle}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
    
    
    def plotHist(data, bins=100, xrange=None):
        """
        Flatten data and plot a histogram with an optional x-axis range.
    
        :param data: Data to be plotted
        :param bins: Number of bins in the histogram
        :param xrange: Tuple specifying the (min, max) range of the x-axis. Default is None.
        """
    
        flattened_data = data.flatten()
    
        plt.hist(flattened_data, bins=bins, range=xrange)
        plt.show()
    
    
    def plotManyHist(data_list, bins=100, xrange=None, cmap='turbo', legend=True):
        """
        Plot a histogram for multiple data arrays with different colors.
    
        :param data_list: List of data arrays to be plotted
        :param bins: Number of bins in the histogram
        :param xrange: Tuple specifying the (min, max) range of the x-axis. Default is None.
        :param cmap: Colormap for different data arrays. Default is 'jet'.
        :param legend: Boolean to specify whether to display the legend. Default is True.
        """
    
        # Define a set of colors
        colors = plt.cm.get_cmap(cmap, len(data_list))
    
        for i, data in enumerate(data_list):
            flattened_data = data.flatten()
            plt.hist(flattened_data, bins=bins, range=xrange, color=colors(i), alpha=0.7, label=f'Array {i+1}')
    
        if legend:
            plt.legend()
    
        plt.show()
        
    def plotHist2(data, bins=100, xrange=None):
        """
        Flatten data and plot a histogram with small text for statistical metrics, including FWHM.
        """
        flattened_data = data.flatten()
    
        # Calculate statistical metrics
        mean = np.mean(flattened_data)
        median = np.median(flattened_data)
        std_dev = np.std(flattened_data)
        skewness = scipy.stats.skew(flattened_data)
        kurtosis = scipy.stats.kurtosis(flattened_data)
        data_range = np.ptp(flattened_data)
    
        # Create histogram and get bin values
        counts, bin_edges = np.histogram(flattened_data, bins=bins, range=xrange)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Find peak of histogram
        peak_idx, _ = hist_peaks(counts)
        if peak_idx.size > 0:
            peak_height = counts[peak_idx[0]]
            half_max = peak_height / 2
    
            # Find nearest points on left and right side of the peak at half maximum
            left_idx = np.where(counts[:peak_idx[0]] < half_max)[0][-1]
            right_idx = np.where(counts[peak_idx[0]:] < half_max)[
                0][0] + peak_idx[0]
    
            # Calculate FWHM
            fwhm = bin_centers[right_idx] - bin_centers[left_idx]
        else:
            fwhm = np.nan  # FWHM is not applicable if no peak is found
    
        # Plot histogram
        plt.hist(flattened_data, bins=bins, range=xrange)
    
        # Add text for metrics
        plt.text(0.03, 0.95, f'Mean: {mean:.4f}\nMedian: {median:.4f}\nStd Dev: {std_dev:.4f}\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}\nRange: {data_range:.4f}\nFWHM: {fwhm:.4f}',
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes, fontsize=14, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
        plt.show()
    
        # Print metrics
        print(f"Metrics: Mean={mean}, Median={median}, Std Dev={std_dev}, Skewness={skewness}, Kurtosis={kurtosis}, Range={data_range}, FWHM={fwhm}")
    
    
    def calculate_fwhm(counts, bin_centers):
        # Find the peak of the histogram
        peak_idx = np.argmax(counts)
        peak_height = counts[peak_idx]
        half_max = peak_height / 2
    
        # Find indices of the half max value
        left_idx = np.where(counts[:peak_idx] <= half_max)[0]
        right_idx = np.where(counts[peak_idx:] <= half_max)[0] + peak_idx
    
        # Ensure indices are found
        if left_idx.size > 0 and right_idx.size > 0:
            left_idx = left_idx[-1]
            right_idx = right_idx[0]
            fwhm = bin_centers[right_idx] - bin_centers[left_idx]
            return fwhm
        else:
            return np.nan
    
    
    def plotHist3(data, bins=100, xrange=None):
        """
        Flatten data and plot a histogram with small text for statistical metrics, including FWHM.
        """
        flattened_data = data.flatten()
    
        # Calculate statistical metrics
        mean = np.mean(flattened_data)
        median = np.median(flattened_data)
        std_dev = np.std(flattened_data)
        skewness = scipy.stats.skew(flattened_data)
        kurtosis = scipy.stats.kurtosis(flattened_data)
        data_range = np.ptp(flattened_data)
    
        # Create histogram and get bin values
        counts, bin_edges = np.histogram(flattened_data, bins=bins, range=xrange)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Calculate FWHM
        fwhm = calculate_fwhm(counts, bin_centers)
    
        # Plot histogram
        plt.hist(flattened_data, bins=bins, range=xrange)
    
        # Add text for metrics
        plt.text(0.03, 0.95, f'Mean: {mean:.4f}\nMedian: {median:.4f}\nStd Dev: {std_dev:.4f}\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}\nRange: {data_range:.4f}\nFWHM: {fwhm:.4f}',
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes, fontsize=14, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
        plt.show()
    
        # Print metrics
        print(f"Metrics: Mean={mean}, Median={median}, Std Dev={std_dev}, Skewness={skewness}, Kurtosis={kurtosis}, Range={data_range}, FWHM={fwhm}")   
    
    def get_phiMapFromGradients(Grad, component, thetaMap):
        
        if component == 'x' or 'X':
            
            phiMap = np.arccos(-Grad/np.tan(thetaMap))
        
        elif component == 'y' or 'Y':
            
            phiMap = np.arcsin(Grad/np.tan(thetaMap))
        
        return phiMap
    
    """Define functions to model the diffraction background"""
    def cartesian_to_polar(y, x, center_y=64, center_x=64):
        # Convert Cartesian coordinates to polar coordinates with respect to the center pixel
        r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        theta = np.arctan2(y - center_y, x - center_x)
        return r, theta
  
    
    def adjust_phi_to_match_gradients(xgrad, ygrad, phiMap):
        """
        Adjust the phi values in phiMap so that the resulting gradients match the 
        gradients given by xgrad and ygrad.
        """
    
        # Convert phiMap to radians
        phi_radians = np.deg2rad(phiMap)
    
        # Calculate gradients from phi
        # Negative tan(theta) is assumed to be tan(90) = infinity
        dx_from_phi = -np.tan(np.pi/2) * np.cos(phi_radians)
        dy_from_phi = np.tan(np.pi/2) * np.sin(phi_radians)
    
        # Identify where the signs differ
        mismatch_x = np.sign(dx_from_phi) != np.sign(xgrad)
        mismatch_y = np.sign(dy_from_phi) != np.sign(ygrad)
    
        # Initialize the mask with zeros (no correction)
        correction_mask = np.zeros_like(phiMap)
    
        # Apply corrections according to the specified rules
        # Both gradients need fixing
        both_mismatch = mismatch_x & mismatch_y
        phi_radians[both_mismatch] += np.pi
        correction_mask[both_mismatch] = 1
    
        # Only x-gradient needs fixing
        x_only_mismatch = mismatch_x & ~mismatch_y
        phi_radians[x_only_mismatch] = np.pi - phi_radians[x_only_mismatch]
        correction_mask[x_only_mismatch] = 2
    
        # Only y-gradient needs fixing
        y_only_mismatch = ~mismatch_x & mismatch_y
        phi_radians[y_only_mismatch] = -phi_radians[y_only_mismatch]
        correction_mask[y_only_mismatch] = 3
    
        phi_radians[phi_radians > 2*np.pi] -= 2*np.pi
        phi_radians[phi_radians < 0] += 2*np.pi
    
        # Convert back to degrees before returning
        corrected_phiMap = np.rad2deg(phi_radians)
    
        return corrected_phiMap, correction_mask
   
    
    def plot_histograms(centers_data):
        """
        Plots separate histograms for all y_coords_centered and x_coords_centered,
        and outputs the standard deviations.
    
        :param centers_data: The dataset with shape (3, A1, A2, 6, 2)
        """
        colors = ['red', 'green', 'blue']  # Define colors for each dataset
    
        # Create figures for y-coords and x-coords histograms
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # y-coords histogram
        plt.title('Histogram of all y_coords_centered')
    
        plt.subplot(1, 2, 2)  # x-coords histogram
        plt.title('Histogram of all x_coords_centered')
    
        # Initialize lists to store standard deviations
        std_dev_y = []
        std_dev_x = []
    
        for i in range(centers_data.shape[0]):
            all_y_coords = []
            all_x_coords = []
    
            for feature_index in range(centers_data.shape[3]):
                # Extract and center the y, x coordinates for the current feature across all images
                y_coords = centers_data[i, :, :, feature_index, 0].flatten()
                x_coords = centers_data[i, :, :, feature_index, 1].flatten()
                y_mean, x_mean = np.mean(y_coords), np.mean(x_coords)
    
                y_coords_centered = y_coords - y_mean
                x_coords_centered = x_coords - x_mean
    
                all_y_coords.extend(y_coords_centered)
                all_x_coords.extend(x_coords_centered)
    
            # Calculate and store standard deviations
            std_dev_y.append(np.std(all_y_coords))
            std_dev_x.append(np.std(all_x_coords))
    
            # Plot histograms
            plt.subplot(1, 2, 1)
            plt.hist(all_y_coords, bins=200, color=colors[i], alpha=0.5, label=f'Dataset {i+1} (std: {std_dev_y[-1]:.2f})', density=True)
    
            plt.subplot(1, 2, 2)
            plt.hist(all_x_coords, bins=200, color=colors[i], alpha=0.5, label=f'Dataset {i+1} (std: {std_dev_x[-1]:.2f})', density=True)
    
        # Add labels and legend
        plt.subplot(1, 2, 1)
        plt.xlabel('y_coords_centered')
        plt.ylabel('Frequency')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.xlabel('x_coords_centered')
        plt.ylabel('Frequency')
        plt.legend()
    
        plt.show()
    
        # Output the standard deviations
        for i in range(len(std_dev_x)):
            print(f"Dataset {i+1}: Std Dev in x_coords_centered = {std_dev_x[i]:.2f}, Std Dev in y_coords_centered = {std_dev_y[i]:.2f}")
    
    
    def plot_all_centers_of_mass(centers_ws2, transparency=0.5, labels=None, drawConvexHull=True):
        """
        Plots the centers of mass for all features from each dataset, centered at (0, 0).
    
        :param centers_ws2: The dataset with shape (3, A1, A2, 6, 2)
        """
        # Define colors for each dataset
        colors = ['red', 'green', 'blue']  # Adjust as needed for the number of datasets
    
        # Create the plot
        plt.figure(figsize=(10, 10))
    
        for i in range(centers_ws2.shape[0]):
            all_coords = []  # To store all coordinates for convex hull
    
            for feature_index in range(centers_ws2.shape[3]):
                # Extract and center the y, x coordinates for the current feature across all images
                y_coords = centers_ws2[i, :, :, feature_index, 0].flatten()
                x_coords = centers_ws2[i, :, :, feature_index, 1].flatten()
                y_mean, x_mean = np.mean(y_coords), np.mean(x_coords)
                y_coords_centered = y_coords - y_mean
                x_coords_centered = x_coords - x_mean
                coords = np.column_stack((x_coords_centered, y_coords_centered))
                all_coords.append(coords)
    
                # Scatter plot for each feature of the dataset
                plt.scatter(x_coords_centered, y_coords_centered, color=colors[i], alpha=transparency)
                    
            
            # Draw convex hull for all features in the dataset
            if drawConvexHull:
                all_coords = np.vstack(all_coords)  # Combine all features' coordinates
                if len(all_coords) > 2:  # Convex hull requires at least 3 points
                    hull = ConvexHull(all_coords)
                    for simplex in hull.simplices:
                        plt.plot(all_coords[simplex, 0], all_coords[simplex, 1], color=colors[i], linewidth=2)
    
            # Add label for the dataset
            if labels is not None:
                plt.plot([], [], color=colors[i], label=labels[i])
            else:
                plt.plot([], [], color=colors[i], label=f'Dataset {i+1}')
    
        # Adding labels, title, and legend
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.title('Centers of Mass for All Features')
        plt.legend()
    
        plt.show()
    
    # def plot_centers_of_mass_with_histograms(centers_data, colors, labels=None, drawConvexHull=True,
    #                                          transparency=0.5, hist_height=0.16, bins=150,
    #                                          alpha=0.5, density=True,):
    #     """
    #     Plots the centers of mass for all features from each dataset, centered at (0, 0),
    #     with histograms of the x and y coordinates.
    
    #     centers_data: The dataset with shape (n_datasets, A1, A2, n_spots, 2)
    #     hist_height: Height of the histograms as a fraction of total figure height
    #     """
        
    #     # Check data compatibility
    #     assert type(drawConvexHull) is bool, "'drawConvexHull' must be a boolean (True/False) variable"
    #     assert type(density) is bool, "'density' must be a boolean  (True/False) variable"
    #     assert len(colors) == centers_data.shape[0], "The number of colors must match the number of datasets to plot."
    #     assert all(isinstance(item, str) for item in colors), "Not all elements are strings."
    
    #     n_datasets = centers_data.shape[0]
    
    #     # Create the main plot
    #     fig = plt.figure(figsize=(10, 10))  
    #     ax_scatter = plt.axes([0.1, 0.1, 0.65, 0.65])
    #     ax_histx = plt.axes([0.1, 0.75, 0.65, hist_height], sharex=ax_scatter)
    #     ax_histy = plt.axes([0.75, 0.1, hist_height, 0.65], sharey=ax_scatter)
    
    #     # Disable labels on histogram to prevent overlap
    #     plt.setp(ax_histx.get_xticklabels(), visible=False)
    #     plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    #     # Initialize standard deviation lists
    #     std_dev_y = np.zeros(n_datasets)
    #     std_dev_x = np.zeros(n_datasets)
    
    #     for i in range(n_datasets):
    #         all_x_coords = []
    #         all_y_coords = []
    
    #         # Collect all coordinates from all feature indices for the current dataset
    #         for feature_index in range(centers_data.shape[3]):
    #             y_coords = centers_data[i, :, :, feature_index, 0].flatten()
    #             x_coords = centers_data[i, :, :, feature_index, 1].flatten()
    #             y_mean, x_mean = np.mean(y_coords), np.mean(x_coords)
    
    #             all_y_coords.extend(y_coords - y_mean)
    #             all_x_coords.extend(x_coords - x_mean)
    
    #             # Scatter plot for each feature of the dataset
    #             ax_scatter.scatter(x_coords - x_mean, y_coords - y_mean, color=colors[i], alpha=transparency)
                
    #         # Calculate and store standard deviations
    #         std_dev_y.append(np.std(all_y_coords))
    #         std_dev_x.append(np.std(all_x_coords))
                
    #         # Combine all x and y coordinates
    #         combined_coords = np.column_stack((all_x_coords, all_y_coords))
    
    #         # Draw convex hull for the combined coordinates of the dataset
    #         if drawConvexHull and len(combined_coords) > 2:
    #             hull = ConvexHull(combined_coords)
    #             for simplex in hull.simplices:
    #                 ax_scatter.plot(combined_coords[simplex, 0], combined_coords[simplex, 1], color=colors[i], linewidth=2)
    
    #         # Add label for the dataset
    #         if labels is not None:
    #             ax_scatter.plot([], [], color=colors[i], label=labels[i])
    #         else:
    #             ax_scatter.plot([], [], color=colors[i], label=f'Dataset {i+1}')
    
    #         # Plot histograms
    #         ax_histx.hist(all_x_coords, bins=bins, color=colors[i], alpha=alpha, 
    #                       density=density, label=rf'$\sigma$ = {std_dev_x[-1]:.2f}')
    #         ax_histy.hist(all_y_coords, bins=bins, color=colors[i], alpha=alpha, orientation='horizontal', 
    #                       density=density, label=rf'$\sigma$ = {std_dev_y[-1]:.2f}')
    
    
    #     # Set labels and title for the scatter plot
    #     ax_scatter.set_xlabel('kx')
    #     ax_scatter.set_ylabel('ky')
    #     ax_scatter.set_title('Centers of Mass with Histograms')
    #     ax_scatter.legend()
    
    #     plt.show()
    
    def apply_majority_filter(cluster_map, kernel_size=3, iterations=1):
        
        """
        Apply a majority filter to a 2D cluster map.
    
        cluster_map: A 2D numpy array with cluster indices.
        kernel_size: The length of the square kernel (must be an odd number).
        
        Returns a 2D numpy array after applying the majority filter.
        """
    
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
    
        def majority_filter(pixel_values):
            # Return the mode (most common element) of the array.
            return mode(pixel_values, axis=None)[0][0]
    
        # Generate a footprint for the filter (square kernel)
        footprint = np.ones((kernel_size, kernel_size))
           
        # Apply generic filter with the custom majority function
        filtered_map = ndimage.generic_filter(
            cluster_map, majority_filter, footprint=footprint, mode='constant', cval=0)
        
        # Filter more times if user specified this 
        if iterations > 1:
            for i in range(iterations-1):
                filtered_map = ndimage.generic_filter(
                    filtered_map, majority_filter, footprint=footprint, mode='constant', cval=0)
        
        return filtered_map
    
    
    def argmedian(data, axis=0):
        sorted_indices = np.argsort(data, axis=axis)
        mid_index = data.shape[axis] // 2
        if data.shape[axis] % 2 == 0:
            # For an even number of elements, choose the lower median's index
            # Creating an index array with the correct shape
            idx_shape = list(data.shape)
            idx_shape[axis] = 1  # Set the median axis to have a size of 1
            median_indices_lower = np.take_along_axis(sorted_indices, np.full(idx_shape, mid_index - 1, dtype=int), axis=axis)
            # You could choose one or average the indices/values here
            median_indices = median_indices_lower  # Or some logic to handle even-sized arrays
        else:
            idx_shape = list(data.shape)
            idx_shape[axis] = 1  # Set the median axis to have a size of 1
            median_indices = np.take_along_axis(sorted_indices, np.full(idx_shape, mid_index, dtype=int), axis=axis)
        return np.squeeze(median_indices, axis=axis)  # Remove the added dimension
    
    
    def get_average_clusters(dataset, cluster_map, plot_averages=True,vmin=4,vmax=14):
        
        A, B, C, D = dataset.shape
        E, F = cluster_map.shape
        
        assert A == E and B == F, "The 3rd and 4th dimensions of 'dataset' must match the dimensions of 'cluster_map'" 
        
        # Count number of clusters
        n_clusters = len(np.unique(cluster_map))
        average_values = np.zeros((n_clusters, C, D))
        
        # Loop over each cluster
        for cluster_idx in range(n_clusters):
                
            # Calculate the average across all selected patterns for this cluster
            average_values[cluster_idx] = np.mean(dataset[cluster_map == cluster_idx], axis=0)
            
        if plot_averages:
            for i in range(n_clusters):
                plotDP(average_values[i],vmin=vmin,vmax=vmax)
        
        return average_values
        
    def get_SNR(dataset, signalMask, resBgMask, noiseMask, returnArray=False):
        """
        Return a signal-to-noise ratio (SNR) measurement for the input 4D dataset or diffraction pattern
        """
        assert len(dataset.shape) == 2 or len(dataset.shape) == 4, "Input 'dataset' must be 2D or 4D"
        
        # Apply masks to data
        dataset_signalMask = dataset*(signalMask - resBgMask)
        dataset_stdMask = dataset*noiseMask
    
        # Single diffraction pattern
        if len(dataset.shape) == 2:
            # Collect average spot intensity
            meanSignal = np.sum(dataset_signalMask)/6
            
            # Collect standard deviation of noisy region
            stdNoise = np.std(dataset_stdMask[dataset_stdMask > 0])
            
            # Compute SNR
            SNR = 20*np.log10(meanSignal/stdNoise)
            
            return SNR
                    
        elif len(dataset.shape) == 4:
            Ry, Rx, Ky, Kx = dataset.shape
            
            # Initialize SNR array
            SNR = np.zeros((Ry, Rx))
            
            
            for i in range(Ry):
                for j in range(Rx):
                                    
                    # Collect average spot intensity
                    meanSignal = np.sum(dataset_signalMask[i, j])/6
                    
                    # Collect standard deviation of noisy region
                    stdNoise = np.std(dataset_stdMask[i, j][dataset_stdMask[i,j ]> 0])
                    
                    # Compute SNR
                    SNR[i, j] = 20*np.log10(meanSignal/stdNoise)
            
            if not returnArray:
                meanSNR = np.mean(SNR)
                stdSNR = np.std(SNR)
                
                return meanSNR, stdSNR
            
    def get_noise(dataset, noiseMask):
        
        assert len(dataset.shape) == 2 or len(dataset.shape) == 4, "Input 'dataset' must be 2D or 4D"
        
        # Apply masks to data
        dataset_stdMask = dataset*noiseMask
    
        # Single diffraction pattern
        if len(dataset.shape) == 2:
            
            # Collect standard deviation of noisy region
            stdNoise = np.std(dataset_stdMask[dataset_stdMask > 0])
            
            return stdNoise
                    
        elif len(dataset.shape) == 4:
            Ry, Rx, Ky, Kx = dataset.shape
            
            # Initialize SNR array
            Noise = np.zeros((Ry, Rx))
            
            
            for i in range(Ry):
                for j in range(Rx):
                                                    
                    # Collect standard deviation of noisy region
                    stdNoise = np.std(dataset_stdMask[i, j][dataset_stdMask[i,j ]> 0])
                    
                    # Compute Noise
                    Noise[i, j] = stdNoise
            
            meanNoise = np.mean(Noise)
            stdNoise = np.std(Noise)
            
            return meanNoise, stdNoise    
    
    def get_full_r_theta_transform(diff, center=None):
        
        height, width = diff.shape
        
        if center is None:
            center = (height // 2, width // 2)
    
        # Generate Cartesian coordinate grid for the original image
        x, y = np.meshgrid(np.arange(width), np.arange(height))
    
        # Flatten the coordinate grids and values for interpolation
        points = np.vstack((x.ravel(), y.ravel())).T
        values = diff.ravel()
    
        # Generate polar coordinate grid (r, theta)
        theta = np.linspace(-np.pi, np.pi, width)  # full 2pi range
        r = np.linspace(0, min(center), height)  # radial range up to the image center
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Convert polar coordinates back to Cartesian for interpolation
        x_grid = center[1] + r_grid * np.cos(theta_grid)
        y_grid = center[0] + r_grid * np.sin(theta_grid)
    
        # Interpolate using the original points and values onto the new grid
        # polar_diff = griddata(points, values, (x_grid, y_grid), method='cubic').reshape(theta_grid.shape)
        polar_diff = griddata(points, values, (x_grid, y_grid), method='cubic').reshape((theta_grid.shape))
    
        return clip_values(polar_diff.T)
    
    def get_polar_data(dataset, center=None):
       
        modified_Dataset = np.zeros_like(dataset)
        
        if len(dataset.shape == 4):
        
            A, B, C, D = dataset.shape
                  
            if center is None:
                center = (C//2, D//2)
    
            for i in range(A):
                for j in range(B):
                    modified_Dataset[i, j] = get_full_r_theta_transform(dataset[i, j], center=center)
                
                print(f"Finished processing row {i+1}/{A}.")
        
        elif len(dataset.shape == 3):
        
            B, C, D = dataset.shape              
    
            if center is None:
                center = (C//2, D//2)
    
            for j in range(B):
                modified_Dataset[j] = get_full_r_theta_transform(dataset[j], center=center)
                
                if j%100 == 0:
                   print(f"Finished processing {j}/{B} diffractions.")
        
        return modified_Dataset
    
    
    def subtract_backgrounds(data, cluster_map, backgrounds, n_clusters):
        """
        Subtract backgrounds from each cluster in the dataset based on the cluster map.
    
        data (numpy.ndarray): Original 4D dataset of shape (A, B, C, D).
        cluster_map (numpy.ndarray): 2D array of shape (A, B) where each value 
                                     indicates the cluster index.
        backgrounds (numpy.ndarray): Array of shape (n_clusters, C, D) containing the 
                                     backgrounds to be subtracted.
    
        Returns the modified dataset after background subtraction as numpy array.
        """
        # Ensure the shapes are as expected
        assert data.shape[:2] == cluster_map.shape
        assert len(backgrounds.shape) == 3
    
        # Initialize the modified data array
        modified_data = data.copy()
    
        # Iterate over each cluster and apply the corresponding background subtraction
        for cluster_idx in range(n_clusters):
            # Find indices of the current cluster
            cluster_indices = np.where(cluster_map == cluster_idx)
    
            # Subtract the background for each pixel in the cluster
            for x, y in zip(*cluster_indices):
                modified_data[x, y] -= backgrounds[cluster_idx]
    
        return modified_data
    
    def get_cluster_masks(clusterMap):
        """
        Separate the clusters in cluster map
        """
        
        A, B = clusterMap.shape
        labels = np.unique(clusterMap)
    
        cluster_masks = np.zeros((len(labels),A,B), dtype=bool)
        
        for label in labels:
            
            cluster_masks[label][clusterMap == label] = clusterMap[clusterMap == label]
            
        return cluster_masks
    
    #%%
    
    rippleData = np.load('rawRippleData.npy')
    singleRipple = rippleData[80:190, 60:100]
    
    np.save('singleRipple.npy', singleRipple)
    
    del(rippleData)
    HyperData = HyperData(singleRipple)
    
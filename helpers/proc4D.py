"""
A set of functions to process and visualize 4D-STEM data

Author: Adan J Mireles
Rice University
April 2024
"""

if __name__ == '__main__':
    import os
    # os.chdir("/home/han/Users/adan/pythonCodes/InverseFunction")
    os.chdir("C:/Users/haloe/Documents/CodeWriting/pythonCodes/HanLab/InverseFunction/")
    import sys
    sys.path.append("C:/Users/haloe/Documents/CodeWriting/4Denoise/")

import numpy as np
import h5py

from scipy.stats import mode
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.signal import find_peaks
import scipy.stats
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter
from scipy import io
from scipy.linalg import polar
from scipy.fft import fft2, fftshift
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
from Node_class import *
# import helper_function as hf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import perf_counter
from tqdm import tqdm
import time
import xgboost as xgb
from genData_Direct import VectorSol_all, VectorSol_matrix
from matplotlib.colors import ListedColormap

from skimage.measure import profile_line
from skimage import transform
from skimage import feature
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
import inspect


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
    
    data = io.loadmat(filename)
    
    for key in data.keys():
        content = data[key]
        
        if isinstance(content, np.ndarray):
            if len(content.shape) > 1:
                return content
    

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

def make_mask(center, r_mask, mask_dim=(128,128)):
    """
    Create a 2D mask of user-defined dimensions at a defined radius from a central point
    """
    
    mask = np.zeros(mask_dim)
    y, x = center
    for i in range(mask_dim[0]):
        for j in range(mask_dim[1]):
            if (i-y)**2 + (j-x)**2 <= r_mask**2:
                mask[i][j] = 1

    return mask

def rem_negs(array):
    """
    Function to replace negative values in diffraction data.
    
    Input: 4D dataset
    Output: 4D dataset with values less than 1 replaced with 1
    """

    array[array < 1] = 1

    return array

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

# def calculate_PSNR(noisy_arr, clean_arr):
#     """
    
#     """
    
#     PSNR = 0

#%% Denoising Functions and Classes


class DenoisingMethods:
    
    # =============================================================================
    # Spatial Filters
    # =============================================================================

    def gaussian(self, target_data, kernel_size=(5, 5), sigma=1):
        """Apply a Gaussian filter to 2D data.
    
        The Gaussian filter reduces noise by averaging the pixel values within a Gaussian kernel,
        creating a smooth image that minimizes high-frequency noise while preserving edges to some extent.
    
        Parameters
        ----------
        target_data : ndarray
            The 2D data to be denoised.
        kernel_size : tuple of int, optional
            The size of the Gaussian kernel (default is (5, 5)).
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
        
        return cv2.GaussianBlur(target_data, kernel_size, sigma)
    
    
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
            Filter sigma in the coordinate space (default is 75).
    
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
    
        return rem_negs(modified_data)

    
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
    
# =============================================================================
# 
# =============================================================================

    #
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
        
        padded_target_data = self._pad_image(target_data, sMax//2)
        H, W = target_data.shape
        filtered_target_data = np.zeros_like(target_data)

        for i in range(H):
            for j in range(W):
                value = self._process_pixel(padded_target_data, i + sMax//2, j + sMax//2, s, sMax)
                filtered_target_data[i, j] = value

        return filtered_target_data
    
    # Private method (helper function) for the adaptive median filter
    def _pad_image(self, image, pad_width):
        """
        Pads the image with the minimum value to handle edge cases.

        Parameters
        ----------
        image : numpy.ndarray
            The image to pad.
        pad_width : int
            The amount of padding to apply to each border.

        Returns
        -------
        padded_image : numpy.ndarray
            The padded image.

        Notes
        -----
        This method uses constant mode padding with the image's minimum as the pad value.
        """
        
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=np.min(image))
        
        return padded_image

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
        
        
        
    #     return rem_negs(filtered_data)    
    
    # # =============================================================================
    # # Tensor Decomposition and Factorization
    # # =============================================================================
    
    # #
    # def pca_sk(self, other):
        
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def kernelPCA_sk(self, other):
        
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def fastICA_sk(self, other):
        
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def nmf_sk(self, other):
        
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def tucker_TL(self, other):
    #     """
    #     Tucker Decomposition based on TensorLy implementation.
    #     [link].
        
    #     """        
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def tucker_zhang(self, other):
    #     """
    #     Tucker Decomposition based on implementation by Zhang et al. (2020).
    #     See reference:
    #         Zhang, C., Han, R., Zhang, A. R., & Voyles, P. M. (2020). 
    #         "Denoising atomic resolution 4D scanning transmission electron 
    #         microscopy data with tensor singular value decomposition." 
    #         Ultramicroscopy, 219, 113123.
        
    #     """
        
    #     # Initial parameters Tucker decomposition in ref.: 7 for real space and 30 for unwrapped k
        
    #     return rem_negs(filtered_data)
    
    
    # #
    # def cp_TL(self, other):
    #     """
    #     CANDECOMP-PARAFAC Decomposition based on TensorLy implementation.
    #     [link]
    #     """
        
        
    #     return rem_negs(filtered_data)

    # #
    # def tensor_train_TL(self, other):
    #     """
    #     Tensor Train Decomposition based on TensorLy implementation.
    #     [link]
    #     """
        
        
    #     return rem_negs(filtered_data)
        
    # #
    # def parafac2_TL(self, other):
    #     """
    #     PARAFAC-2 Decomposition based on TensorLy implementation.
    #     [link]
    #     """
        
        
    #     return rem_negs(filtered_data)
    
    # #
    # def partialTucker_TL(self, other):
    #     """
    #     Partial Tucker Decomposition based on TensorLy implementation.
    #     [link]
    #     """
        
        
    #     return rem_negs(filtered_data)
    
    
    # # =============================================================================
    # # Transform Domain Filtering
    # # =============================================================================
    
    
    # #
    # def fourier_filt(self, other):
    #     """
    #     Fourier filtering.
        
    #     If 'apply_log' is True, we obtain the exit wave power cepstrum (EWPC)
    #     discussed in [reference].
    #     """
        
        
    #     return rem_negs(filtered_data) 
    
    # #
    # def wavelet_thresholding(self, other):
    #     """
    #     Fourier filtering.
        
    #     If 'apply_log' is True, we obtain the exit wave power cepstrum (EWPC)
    #     discussed in [reference].
    #     """
        
        
    #     return rem_negs(filtered_data) 
    
    # #
    # def curvelet(self, other):
    #     """
        
    #     """
        
    #     return rem_negs(filtered_data) 
    
    # # 
    # def block_matching(self, other):
    #     """
    #     Block-matching and 3D/4D Filtering (BM3D or BM4D)
        
    #     Input data must be two- or three-dimensional.
        
    #     Based on [reference]
    #     """
        
    #     return filtered_data
        
    
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
        self.target_data = target_data
        self.methods = DenoisingMethods()

    def denoise(self, method_name, target_data=None, **kwargs):
        
        if target_data is None:
            target_data = self.target_data    

        method = getattr(self.methods, method_name, None)

        if method:
            
            try:
                return method(target_data, **kwargs)
            
            except TypeError as e:
                sig = inspect.signature(method)
                param_names = ', '.join([param.name for param in sig.parameters.values() if param.name != 'target_data'])
                raise TypeError(f"{str(e)}. Valid arguments are: {param_names}")

        else:
            available_methods = [m for m, v in inspect.getmembers(self.methods, predicate=inspect.ismethod) if not m.startswith('__')]
            raise ValueError(f"No such method '{method_name}'. Available methods are: {', '.join(available_methods)}")

    def apply_denoising(self, real_space_method=None, real_space_kwargs=None, 
                        reciprocal_space_method=None, reciprocal_space_kwargs=None, **kwargs):
        
        # =============================================================================
        # Spatial Domain Filtering
        # =============================================================================
        
        if real_space_method is None and reciprocal_space_method is None:
            
            # TODO: Will need to assume here that method could be non-spatial
            raise ValueError("No denoising method provided.")
                
        if real_space_kwargs is None:
            real_space_kwargs = {}
        if reciprocal_space_kwargs is None:
            reciprocal_space_kwargs = {}
        if real_space_method:
            real_space_kwargs.update(kwargs)
        if reciprocal_space_method:
            reciprocal_space_kwargs.update(kwargs)
        
        dims = self.target_data.ndim

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
                return np.array([self.denoise(real_space_method, target_data=slice, **real_space_kwargs) for slice in self.target_data])
            elif reciprocal_space_method and not real_space_method:
                return np.array([self.denoise(reciprocal_space_method, target_data=slice, **reciprocal_space_kwargs) for slice in self.target_data])
            elif real_space_method and reciprocal_space_method:
                raise ValueError("Cannot handle both real space and reciprocal space methods simultaneously for 3D data.")

        elif dims == 4:
            ry, rx, ky, kx = self.target_data.shape
            processed_data = np.zeros_like(self.target_data)

            if real_space_method:
                for i in range(ky):
                    for j in range(kx):
                        processed_data[:, :, i, j] = self.denoise(real_space_method, target_data=self.target_data[:, :, i, j], **real_space_kwargs)

            if reciprocal_space_method:
                for i in range(ry):
                    for j in range(rx):
                        processed_data[i, j, :, :] = self.denoise(reciprocal_space_method, target_data=self.target_data[i, j, :, :], **reciprocal_space_kwargs)

            return HyperData(processed_data)

        else:
            raise ValueError("Unsupported image dimensionality")

#%%

class HyperData:
    
    def __init__(self, data):
        self.data = data
        self.denoiser = Denoiser(data)

    def swap_coordinates(self):
        """
        Swap the real space and reciprocal space coordinates in the 4D dataset.
        For a dataset with dimensions (A, B, C, D),
        this method swaps them to (C, D, A, B).
        
        Returns:
        - swapped_data (numpy.ndarray): The 4D dataset with swapped dimensions.
        """
        
        # The original order is (0, 1, 2, 3) and we want to change to (2, 3, 0, 1)
        swapped_data = np.transpose(self.data, (2, 3, 0, 1))
        
        return HyperData(swapped_data)
    
    def alignment(self, r_mask=5):
        """
        Align the diffraction patterns through the Center of mass of the center beam
        
        Function written by Chuqiao Shi (2022)
        See on GitHub: Chuqiao2333/Hierarchical_Clustering
        
        Modified by Adan J Mireles (April, 2024)
        - Reordered dimensions (x, y) to (y, x)
        """

        y, x, ky, kx = np.shape(self.data)
        data_type = self.data.dtype  # Store the original data type
        com_y, com_x = self._quickCOM(r_mask=r_mask) 
        cbed_tran = np.zeros((y, x, ky, kx), dtype=data_type)
        
        print(f'Processing {y} by {x} real-space positions...')        
        for i in tqdm(range(y), desc = 'Alignment Progress'):
            for j in range(x):
                afine_tf = transform.AffineTransform(translation=(-ky//2+com_y[i,j], -kx//2+com_x[i,j]))
                cbed_tran[i,j,:,:] = transform.warp(self.data[i,j,:,:], inverse_map=afine_tf)
                # sys.stdout.write('\r %d,%d' % (i+1, j+1) + ' '*10)
        
        cbed_tran_Obj = HyperData(cbed_tran)
        com_y2, com_x2 = cbed_tran_Obj._quickCOM()
        std_com = (np.std(com_y2), np.std(com_x2))
        mean_com = (np.mean(com_y2), np.mean(com_x2))
        
        print()
        print(f'Standard deviation statistics (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
        print(f'COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')
        
        return cbed_tran_Obj, mean_com, std_com
    
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
        
        y, x, ky, kx = np.shape(self.data)
        center_x = kx // 2
        center_y = ky // 2

        # Assuming make_mask function is defined elsewhere that creates a circular mask
        mask = make_mask((center_y, center_x), r_mask, mask_dim=(ky, kx))
        
        ap2_x = np.zeros((y, x))
        ap2_y = np.zeros_like(ap2_x)
        vx = np.arange(kx)
        vy = np.arange(ky)

        for i in tqdm(range(y), desc = 'Computing centers of mass'):
            for j in range(x):
                cbed = np.squeeze(self.data[i, j, :, :] * mask)
                pnorm = np.sum(cbed)
                if pnorm != 0:
                    ap2_y[i, j] = np.sum(vy * np.sum(cbed, axis=0)) / pnorm
                    ap2_x[i, j] = np.sum(vx * np.sum(cbed, axis=1)) / pnorm

        return ap2_y, ap2_x
    
    # Must add type of interpolation
    def fix_elliptical_distortions(self, r=None, R=None, interp_method='linear', return_fix=True):
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
        
        A, B, C, D = self.data.shape
        
        if not r:
            r = C//5 
        if not R:
            R = 2*C//5
        
        mean_pattern = self.get_dp(special = 'mean').data
        params = self._extract_ellipse(mean_pattern, C//2, D//2, r, R)
        corrected_data = np.empty_like(self.data)
        
        Ang, a, b = params 
        print(f"Ellipse rotation = {np.degrees(Ang)} degrees \nMajor axis 'a' = {a} px \nMinor axis 'b' = {b} px")
        
        if return_fix:
        
            for i in range(A):
                for j in range(B):
                    corrected_data[i, j] = self._apply_affine_transformation(self.data[i, j], *params, interp_method)
    
            return HyperData(corrected_data)

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
    
    def centerBeam_Stats(self, square_side=8):
        """
        Analyze diffraction patterns to find the mean and standard deviation of the center of mass.
    
        Parameters:
        data (numpy.ndarray): 4D dataset of shape (A, B, C, D).
        square_side (int): Side length of the square used to calculate the center of mass.
    
        Returns:
        tuple: Mean and standard deviation of the center of mass coordinates.
        """
        A, B, C, D = self.data.shape
        com_coordinates = []
    
        # Define the region for center of mass calculation
        half_side = square_side // 2
        center = C//2
        min_coord, max_coord = center - half_side, center + half_side
    
        # Calculate center of mass for each BxB image
        for i in range(A):
            for j in range(B):
                region = self.data[i, j, min_coord:max_coord, min_coord:max_coord]
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
    
    def get_stdDev(self, space='reciprocal'):
        """
        Calculate a 2D mask of the standard deviation of each reciprocal space 
        pixel across all diffraction patterns.

        Parameters:
        data (numpy.ndarray): 4D STEM dataset

        Returns:
        numpy array: 2D mask of standard deviations with same shape as diffraction pattern.
        """
        # Validate the shape of the data
        if len(self.data.shape) != 4:
            raise ValueError("Data must be a 4D array")
                
        if space == 'reciprocal':
            # Calculate the standard deviation for each pixel across all diffraction patterns
            std_dev = np.std(self.data, axis=(0, 1))
        elif space == 'real':
            # Calculate the standard deviation for each pixel across all scanning positions
            std_dev = np.std(self.data, axis=(2, 3))
        else:
            raise ValueError("'space' must be 'reciprocal' or 'real'")
            
        return std_dev
    
    def get_dp(self, y=None, x=None, special=None):
        """
        Obtain a diffraction pattern from a specific 4D dataset, either at a 
        specific point, averaged over a specified region, or a special diffraction
        such as the dataset's 'mean', 'median', 'max', etc..
        
        Inputs:
            y: (tuple of int or int, optional) If tuple (ymin, ymax), it 
               specifies the vertical coordinate range to average over. If int,
               it specifies a specific vertical coordinate.
            x: (tuple of int or int, optional) If tuple (xmin, xmax), it 
               specifies the horizontal coordinate range to average over. If 
               int, it specifies a specific horizontal coordinate.
        
        Returns: 
            dp: (numpy array) Diffraction pattern, either at a specific point or averaged over a region.
        
        Raises:
            ValueError: If inputs are improperly specified or not provided.
        """
        
        operations = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'random': np.random.random,
        }

        if special:
            if special in operations:
                if special == 'random':
                    ky = round(self.data.shape[0]*operations[special]())
                    kx = round(self.data.shape[1]*operations[special]())
                    result = self.data[ky, kx]
                else:
                    result = operations[special](self.data, axis=(0, 1))
                return ReciprocalSection(result)
            else:
                valid_operations = ', '.join(f"'{op}'" for op in operations.keys())
                raise ValueError(f"'special' must be a string: {valid_operations}.")
       
        if y or x:
            
            if isinstance(y, tuple) and isinstance(x, tuple):
                ymin, ymax = y
                xmin, xmax = x
                return ReciprocalSection(np.mean(self.data[ymin:ymax, xmin:xmax, :, :], axis=(0, 1)))
            
            elif isinstance(y, tuple) and isinstance(x, int):
                ymin, ymax = y
                return ReciprocalSection(np.mean(self.data[ymin:ymax, x, :, :], axis=0))
            
            elif isinstance(y, int) and isinstance(x, tuple):
                xmin, xmax = x
                return ReciprocalSection(np.mean(self.data[y, xmin:xmax, :, :], axis=0))
            
            elif isinstance(y, int) and isinstance(x, int):
                return ReciprocalSection(self.data[y, x])
            
            else:
                raise ValueError("y and x must be either both tuples, both integers, or one tuple and one integer.")
    
        else:
            raise ValueError("No parameters specified.")

    def bin4D(self):
        """
        Bin 4D dataset by averaging over the first two dimensions.
        """
        
        assert len(self.data.shape) == 4, "'dataset' must be four-dimensional"
        
        A, B, C, D = self.data.shape[0] // 2, self.data.shape[1] // 2, self.data.shape[2], self.data.shape[3]
        binned_dataset = np.zeros((A, B, C, D))

        for i in range(A):
            for j in range(B):
                # Average over the first two dimensions
                binned_dataset[i, j] = np.mean(self.data[2*i:2*i+2, 2*j:2*j+2], axis=(0, 1))

        return HyperData(binned_dataset)
    
    def plotVirtualImage(self, r_minor, r_major, vmin=None, vmax=None, 
                         grid=True, num_div=10,plotMask=False,gridColor='black',
                         plotAxes=True,bds=None, returnArrays = False):
        """
        Plot virtual detector image
        - r_minor, r_major specify the dimensions of the annular ring over which to integrate
        - vmin, vmax specify the limiting pixel values to be plotted
        - bds is tuple (y1, y2, x1, x2) used to crop the real-space image
        
        returnArrays: if True, returns the virtual image and detector mask
        """
        
        assert len(self.data.shape)==4, "Input dataset must be 4-dimensional"
        
        A,B,C,D = self.data.shape
        detector_mask = np.zeros((C,D)) # Diffraction pattern mask
        
        # Identify reciprocal space pixels within desired ring-shaped region
        for i in range(C):
            for j in range(D):
                if r_minor<=np.sqrt((i-C/2)**2+(j-D/2)**2)<=r_major:
                    detector_mask[i,j] = 1
        
        masked_image = np.sum(self.data*detector_mask,axis=(2,3))
        
        if vmin:
            vmin = vmin
        else:
            vmin = np.min(masked_image[masked_image>0])
        if vmax:
            vmax = vmax
        else:
            vmax=np.max(masked_image)
            
        plt.imshow(masked_image, vmin=vmin, vmax=vmax, cmap='gray')
    
        if plotAxes:
            plt.axis('on')
            plt.xticks(np.arange(0, B, B//(num_div+1)))
            plt.yticks(np.arange(0, A, A//(num_div+1)))
        else:
            plt.axis('off')    
    
        if grid:
            plt.grid(c=gridColor)
        plt.show()
        
        if plotMask:
            if bds is None:
                plt.imshow(np.log(np.mean(self.data, axis=(0,1))*detector_mask), cmap='turbo', vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.show()
            else:
                y1, y2, x1, x2 = bds
                plt.imshow(np.log(np.mean(self.data[y1:y2, x1:x2], axis=(0,1))*detector_mask), 
                           cmap='turbo', vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.show()
    
        if returnArrays:
            return masked_image, np.mean(self.data, axis=(0,1))*detector_mask
    
    def getAllCenters4Ddata(self, orders=[1,2,3], r=6.6,
                            coords=coords, num_spots_per_order=6, iterations=1):
        """
        Generate an array with all spot centers in 4D dataset
        """
        
        assert len(self.data.shape) == 4, "Input dataset must be 4-dimensional"
        Ny, Nx, _, _ = self.data.shape
        
        # Case 1: user wants to extract all centers of mass for a single order
        if isinstance(orders, int):       
            all_centers = np.zeros((Ny, Nx, num_spots_per_order, 2))
            for i in range(Ny):
                for j in range(Nx):
                    
                    dp = ReciprocalSection(self.data[i][j])
                    
                    all_centers[i][j] = dp.getCenters(coords=coords,
                                                      spots_order=orders, 
                                                      r=r,iterations=iterations)
                    
        # Case 2: two or more spot orders are of interest
        else:    
            if isinstance(orders, (list, np.ndarray, tuple)):
                num_orders = len(orders)  
                all_centers = np.zeros((num_orders, Ny, Nx, num_spots_per_order, 2))
                
                # Iterate over each spot order in the list, tuple, or array provided
                for o_idx, order in enumerate(orders):
                    centers = np.zeros((Ny, Nx, num_spots_per_order, 2))
                    
                    for i in range(Ny):
                        for j in range(Nx):
                            
                            dp = ReciprocalSection(self.data[i][j])
                            
                            if isinstance(r, list):
                                
                                centers[i][j] = dp.getCenters(spots_order=order, 
                                                              coords=coords, r=r[o_idx])
                            else:
                                
                                centers[i][j] = dp.getCenters(spots_order=order, coords=coords,
                                                           r=r)
                            
                    all_centers[order-1] = centers
            else:
                raise TypeError("'orders' must be a list, numpy array, tuple, or integer")
            
        return all_centers

    def remove_center_beam(self, radius, replacement_value=1, outer_ring=None):
        """
        Apply a circular mask to the center of each 2D image in a 4D dataset.
    
        Parameters:
        radius (int or float): Radius of the circular mask.
        replacement_value (int or float): number used as replacement for masked region
        outer_ring (int or float): radius from the center beyond which all values will be replaced
    
        Returns:
        numpy.ndarray: The modified dataset with masks applied.
        """
        
        A1, A2, B1, B2 = self.data.shape
    
        # Create a circular mask
        Y, X = np.ogrid[:B1, :B2]
        center = (B1 // 2, B2 // 2)
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
    
        if outer_ring:
            mask_outer = dist_from_center >= outer_ring
    
        # Apply the mask to each 2D image
        masked_data = self.data.copy()
        for i in range(A1):
            for j in range(A2):
                masked_data[i, j][mask] = replacement_value
    
                if outer_ring:
                    masked_data[i, j][mask_outer] = replacement_value
    
        return HyperData(masked_data)

#%%

class ReciprocalSection:

    def __init__(self, data):
        self.data = data
        self.denoiser = Denoiser(data)
    
    def show(self, power=1, title='Diffraction Pattern', logScale=True, 
             axes=True, vmin=None, vmax=None,figsize=(10,10), aspect=None,cmap='turbo'):
        """
        Visualize a desired diffraction pattern
        
        power: (int or float) 
        """
    
        # Visualize
        plt.figure(figsize=figsize)
            
        if logScale:
            processed_data = power * np.log(self.data)
        else:
            processed_data = self.data ** power
        
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
            
            ky, kx = self.data.shape
            
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
    
    def getSpotCenter(self, ky, kx, r, plotSpot=False, iterations=1):
        """
        Find the center of mass (of pixel intensities) of a diffraction spot, 
        allowing for non-integer radii.
        """
        
        # Pad and round up to ensure the entire radius is accommodated
        pad_width = int(np.ceil(r))
        padded_data = np.pad(self.data, pad_width=pad_width, mode='constant')
    
        # Adjustment of padding coordinates
        ky_padded, kx_padded = ky + pad_width, kx + pad_width
        
        # Determine the size of the area to extract based on r, ensuring it matches the mask's dimensions
        area_size = int(np.ceil(r * 2))
        if area_size % 2 == 0:
            area_size += 1  # Ensure the area size is odd to match an odd-sized mask
            
        for i in range(iterations):
            # Generate the circular mask with the correct dimensions
            mask = circular_mask(area_size // 2, area_size // 2, r)
            
            # Extract the region of interest from the padded data
            ymin, ymax = int(ky_padded - (area_size // 2)), int(ky_padded + (area_size // 2))
            xmin, xmax = int(kx_padded - (area_size // 2)), int(kx_padded + (area_size // 2))
            spot_data = padded_data[ymin:ymax + 1, xmin:xmax + 1]
        
            # Check if shapes match, otherwise adjust
            if spot_data.shape != mask.shape:
                min_dim = min(spot_data.shape[0], mask.shape[0], spot_data.shape[1], mask.shape[1])
                spot_data = spot_data[:min_dim, :min_dim]
                mask = mask[:min_dim, :min_dim]
        
            # Apply mask and calculate its CoM
            masked_spot_data = spot_data * mask
            com_y, com_x = center_of_mass(masked_spot_data)
            
            if plotSpot:
                # Create turbo colormap with 0-values white
                base_cmap = plt.cm.turbo
                custom_cmap = ListedColormap(np.concatenate(([np.array([1, 1, 1, 1])], 
                                                             base_cmap(np.linspace(0, 1, 2**12))[1:]), axis=0))
                plt.imshow(masked_spot_data, cmap=custom_cmap)
                plt.colorbar()
                plt.show()
            
            ky_padded = com_y + ymin
            kx_padded = com_x + xmin
                
        # Adjust the CoM to account for padding
        ky_CoM = ky_padded - pad_width
        kx_CoM = kx_padded - pad_width
    
        return ky_CoM, kx_CoM

    # Needs update
    def masked_DPs(data, mask_radius, coords, order=None, title=None, 
                   plotSpots=False,return_mask=False,iterations=1,plot=True):
        """ 
        Generate masked diffraction plots for each order.
        Order = 1,2,3,4; Last option plots all.
        """
        
        A,B = data.shape
        compound_mask = np.zeros((A, B))
        
        # Obtain CoMs of interst
        centers = getCenters(data, order, mask_radius, coords=coords, plotSpots=plotSpots,iterations=iterations)
    
        for mask_center in centers:
                    
            mask_spot = make_mask(mask_center, mask_radius, mask_dim=(A,B))
            compound_mask = compound_mask + mask_spot     
        
        # Apply compound mask
        masked_data = data*compound_mask
        
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
    
    def getCenters(self, spots_order=None, r=6, coords=coords, num_spots_per_order=6, 
                   plotSpots=False, iterations=1):
        """
        Generate an array spot centers for any DP
        """
        
        assert len(self.data.shape) == 2, "Input data must be of 2-dimensional"
        
        if spots_order is not None:
            
            # Define spot indeces to extract
            spot_adder = (spots_order - 1)*num_spots_per_order
        
            order_centers = np.zeros((num_spots_per_order, 2))
            for j in range(num_spots_per_order):
                order_centers[j] = self.getSpotCenter(coords[j + spot_adder][0],
                                                 coords[j + spot_adder][1],
                                                 r, plotSpots,iterations=iterations)
                
        else:
            
            order_centers = np.zeros((len(coords), 2))
            
            if isinstance(r, (int, float)):
                for j in range(len(coords)):
                    order_centers[j] = self.getSpotCenter(coords[j][0],
                                                         coords[j][1],
                                                         r, plotSpots,iterations=iterations)
                
            else:
                for j in range(len(coords)):
                    order_centers[j] = self.getSpotCenter(coords[j][0],
                                                         coords[j][1],
                                                         r[j], plotSpots,iterations=iterations)
            
        return order_centers


    def getIntensities(self, spots_order=None, center=None, r=6, coords=None):
        """
        Extract Bragg peak intensities from a single DP
        """
    
        # Determine how many intensities are to be extracted
        if type(spots_order) == int:
            if spots_order == 4:
                ints = np.zeros(18)
            elif spots_order < 4:
                ints = np.zeros(6)
        
            if center is None:
                center = self.getCenters(spots_order, r, coords)
        
            for int_idx, intensity in enumerate(ints):
        
                masked_data = self.data*make_mask(center[int_idx], r)
                ints[int_idx] = np.sum(masked_data[int(center[int_idx][0]-(r+1)):int(center[int_idx][0]+(r+1)),
                                                   int(center[int_idx][1]-(r+1)):int(center[int_idx][1]+(r+1))])
    
        elif spots_order is None:
            
            num_ints = len(center)
            ints = np.zeros(num_ints)
            
            if isinstance(r, (int, float)):
                for int_idx in range(num_ints):
                    masked_data = self.data*make_mask(center[int_idx], r)
                    ints[int_idx] = np.sum(masked_data)            
                
            else:
                for int_idx in range(num_ints):
                    masked_data = self.data*make_mask(center[int_idx], r[int_idx])
                    ints[int_idx] = np.sum(masked_data)
        
        return ints
    
    
    def remove_bg(self, min_sigma=0.85, 
                      max_sigma=2.5, 
                      num_sigma=100, 
                      threshold=85,
                      bg_frac=0.99,
                      mask_radial_frac=0.9,
                      min_distance_factor=0.8,
                      max_distance_factor=1.5,
                      search_radius_factor=2.5, # When refining centers of mass
                      iterations=5,
                      n_estimators = 60, learning_rate = 0.1, max_depth = 50, # Predictor model parameters
                      radii_multiplier=1.5, radii_slope=1.38,
                      vmin=4,vmax=14, get_bg = True,
                      axes=False,plotInput=False,plotCenters=True,plotBg=False,plotResult=True,):
        
        A, B = self.data.shape
        mask = make_mask((A//2,B//2), r_mask=(A+B)//4*mask_radial_frac) # Define region where spots are to be found
        
        # Extract centers and radii automatically
        blobs_log = feature.blob_log(self.data*mask, min_sigma=min_sigma, 
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
        min_threshold = np.mean(np.min(distances,axis=0))*min_distance_factor  # Minimum acceptable distance to the nearest neighbor.
        max_threshold = np.mean(np.min(distances,axis=0))*max_distance_factor  # Maximum acceptable distance to the nearest neighbor.
    
        # Filter blobs. Keep a blob if its nearest neighbor is neither too close nor too far.
        filtered_indices = np.where((nearest_neighbor_dist >= min_threshold) 
                                    & (nearest_neighbor_dist <= max_threshold)
                                    & (((blobs_array[:,0]-A//2)**2+(blobs_array[:,1]-B//2)**2)**.5 <= (A+B)//4*mask_radial_frac*(8/9)))[0]
    
        # Apply filter
        blobs_log = np.zeros((len(filtered_indices), 3))
        blobs_log[:, :2] = blobs_array[filtered_indices]
        blobs_log[:, 2] = r_vals[filtered_indices]
        
        
        if plotCenters:
        
            plt.figure(figsize=(10, 10))
            plt.imshow(np.log(self.data), cmap='turbo',vmin=vmin,vmax=vmax)
            plt.axis('off')
        
            # Circle around each blob
            for blob in blobs_log:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                plt.gca().add_patch(c)
        
            plt.show()
        
        # Refine peak centers using CoM
        peak_centers = self.getCenters(r=blobs_log[:,2]*search_radius_factor,
                                       coords=blobs_log[:,:2], iterations=iterations)
        
        distance_to_center = np.linalg.norm(peak_centers-(A+B)//4,axis=1)
        radii = blobs_log[:,2]*radii_multiplier + radii_slope*(1 - distance_to_center/np.max(distance_to_center))
        
        if get_bg:
            # Masks sizes are based on computed radius of spot and distance to center (center spots with greater radius than far spots)
            dp_bg = predict_values(self.data,centers=peak_centers,radius=radii,
                                   n_estimators=n_estimators,learning_rate=learning_rate, max_depth=max_depth)
    
        if plotInput:
            self.show(vmin=vmin,vmax=vmax,axes=axes, title="Input Diffraction Pattern")
        
        if plotBg and get_bg:
            ReciprocalSection(dp_bg).show(vmin=vmin,vmax=vmax,axes=axes)
    
        if plotResult:
            ReciprocalSection(rem_negs(self.data-bg_frac*dp_bg)).show(vmin=vmin,vmax=vmax,axes=axes,
                   title=f'ne={n_estimators},learning_rate={learning_rate}, max_depth={max_depth}')
        
        if get_bg:
            return peak_centers, radii, dp_bg
        
        else:
            return peak_centers, radii

    
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
    rippleData = np.load('rawRippleData.npy')
    
    """
    Approximate center of first thru third order spots
    
    The coordinates were estimated based on approximate location of spots observed
    within average diffraction pattern. 
    """
    
    #%%
    
    singleRipple = np.load('singleRipple.npy')
    ripple_bg = np.load('ripple_bg.npy')
    rippleData = HyperData(rem_negs(singleRipple - 0*ripple_bg))
    rippleData = rippleData.alignment(7)[0]
    # rippleData_noBg = HyperData(rem_negs(singleRipple - 0.9*ripple_bg))
    
    # Available methods are: anisotropic_diffusion (needs redefining), bilateral, gaussian, median, 
    #                        non_local_means, total_variation
    
    # HyperData_denoised = HyperData.denoiser.apply_denoising(real_space_method='total_variation',
    #                                                   real_space_kwargs={'weight':30, 'eps': 0.0001, 'max_num_iter': 100})
    
    # 46 +- 3
    corrected_rippleData = rippleData.fix_elliptical_distortions(r=42,R=50, interp_method='linear')
    
    rippleData.get_dp(special='mean').show(vmin=4,vmax=14,title="Mean Diffraction Before Elliptical Distortion Correction")
    corrected_rippleData.get_dp(special='mean').show(vmin=4,vmax=14,title="Mean Diffraction After Elliptical Distortion Correction")
    plt.imshow(rippleData.get_dp(special='mean').data - corrected_rippleData.get_dp(special='mean').data,cmap='RdBu',vmin=-1000,vmax=1000)
    
    corrected_rippleData = corrected_rippleData.fix_elliptical_distortions(r=45,R=49,  interp_method='linear')
    
    corrected_rippleData = HyperData(corrected_rippleData.data[35:75])
    
    # import time
    # start = time.perf_counter()
    # rippleData_denoised = rippleData.denoiser.apply_denoising(real_space_method='adaptive_median_filter',
    #                                                   real_space_kwargs={'s':3, 'sMax': 7})
    # end = time.perf_counter()
    # print("It took ", end-start, " seconds." )
    
    # np.save('singleRipple_denoised_noBg.npy', rippleData_denoised.data)
    
    
    #%%
    # HyperData_denoised = corrected_rippleData.denoiser.apply_denoising(real_space_method='anisotropic_diffusion',
    #                                                    real_space_kwargs={'niter':10, 'kappa': 30, 'gamma': 0.25, 'option': 2})
                                                      # niter=10, kappa=30, gamma=0.25, option=2)
    
    HyperData_denoised = corrected_rippleData.denoiser.apply_denoising(reciprocal_space_method='bilateral',real_space_method='anisotropic_diffusion',
                                                                       real_space_kwargs={'niter':10, 'kappa': 30, 'gamma': 0.25, 'option': 2},
                                                                       reciprocal_space_kwargs={'d':5, 'sigma_color': 150, 'sigma_space': 100})
    # (reciprocal_space_method='anisotropic_diffusion',
    #                                                    reciprocal_space_kwargs={'niter':10, 'kappa': 30, 'gamma': 0.25, 'option': 2})
    
    
    corrected_rippleData.plotVirtualImage(40,60,plotAxes=False)
    HyperData_denoised.plotVirtualImage(40,60,plotAxes=False)
    
    corrected_rippleData.get_dp(20,20).show(vmin=4, axes=False, )
    corrected_rippleData.get_dp((18,22),(18,22)).show(vmin=4, axes=False, )

    HyperData_denoised.get_dp(20,20).show(vmin=4, axes=False, )
    
    # HyperData_denoised = HyperData_denoised.denoiser.apply_denoising(reciprocal_space_method='bilateral',
    #                                                   reciprocal_space_kwargs={'d':5, 'sigma_color': 60, 'sigma_space': 90})
    
    
    # HyperData.get_dp(50,20).show(vmin=4, axes=False, )
    # HyperData_denoised.get_dp(50,20).show(vmin=4, axes=False,)
    # HyperData_denoised.plotVirtualImage(40,60,plotAxes=False)
    
    #%%
    
    im, _ = rippleData.plotVirtualImage(20,40,plotAxes=False,returnArrays=True,)
    # im2, _ = HyperData(rippleData_denoised.data[1:-1,1:-1]).plotVirtualImage(40,60,plotAxes=False,vmin=vmin, vmax=vmax, returnArrays=True)
    im2, _ = rippleData_denoised.plotVirtualImage(20,40,plotAxes=False,returnArrays=True)
    
    rippleData.get_dp(60,18).show(vmin=4)
    rippleData_denoised.get_dp(60,18).show(vmin=4)
    
    
    #%%
    
    
    # target_data = im   # Random noise image
    # target_data = rippleData.data[:,:,62,63]   # Random noise image
    target_data = im
    vmin = np.min(target_data)
    vmax = np.max(target_data)
    plt.imshow(target_data, )
    plt.title('original')
    plt.show()
    denoiser = DenoisingMethods()
    
    filtered_target_data = denoiser.adaptive_median_filter(target_data, s=3, sMax=5)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('3,5')
    plt.show()
    
    filtered_target_data = median_filter(target_data, 3)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('3')
    plt.show()
    
    filtered_target_data1 = denoiser.adaptive_median_filter(target_data, s=3, sMax=3)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('3,3')
    plt.show()
    
    filtered_target_data2 = denoiser.adaptive_median_filter(target_data, s=3, sMax=7)
    plt.imshow(filtered_target_data2, vmin=vmin,vmax=vmax)
    plt.title('3,7')
    plt.show()
    
    filtered_target_data = denoiser.adaptive_median_filter(target_data, s=5, sMax=7)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('5,7')
    plt.show()
    
    filtered_target_data = denoiser.adaptive_median_filter(target_data, s=5, sMax=9)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('5,9')
    plt.show()
    
    
    filtered_target_data = denoiser.adaptive_median_filter(target_data, s=7, sMax=9)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('7,9')
    plt.show()
    
    filtered_target_data = denoiser.adaptive_median_filter(target_data, s=7, sMax=11)
    plt.imshow(filtered_target_data, vmin=vmin,vmax=vmax)
    plt.title('7,11')
    plt.show()
    
    np.sum(filtered_target_data2 - filtered_target_data1)
    
    #%% Anisotropic diffusion testing
    
    # Reciprocal space denoising is bad
    # Real space denoising is very similar to full 4D denoising
    
    singleRipple = np.load('singleRipple.npy')
    HyperData = HyperData(rem_negs(singleRipple))
    denoised_real = HyperData.denoiser.apply_denoising(real_space_method='anisotropic_diffusion',
                                                      real_space_kwargs={'niter':15, 'kappa': 25, 'gamma': 0.3, 'option': 2})
    
    denoised_reciprocal = HyperData.denoiser.apply_denoising(reciprocal_space_method='anisotropic_diffusion',
                                                      reciprocal_space_kwargs={'niter':15, 'kappa': 25, 'gamma': 0.3, 'option': 2})
    
    data2denoise = Denoiser(HyperData.data)
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
    
    
    
    #%% Atomic res data
    
    import numpy as np
    import scipy.io
    
    def read_mat_file(filename):
        try:
            data = scipy.io.loadmat(filename)
            # Assuming the variable inside .mat you want is named 'data'
            # You need to replace 'data' with the actual variable name you want to extract.
            # Usually, the variable names can be listed with data.keys()
            if 'data' in data:
                dp = data['data']
                print("Data loaded successfully.")
                return dp
            else:
                print("Data key not found in the .mat file.")
                return data
        except NotImplementedError:
            print("This file may be in HDF5 format. Trying h5py to load.")
            return read_mat_file_h5py(filename)
        except Exception as e:
            print(f"Failed to read the .mat file: {e}")
            return None
    
    def read_mat_file_h5py(filename):
        import h5py
        try:
            with h5py.File(filename, 'r') as file:
                # List all groups
                print("Keys: %s" % list(file.keys()))
                # Let's assume data is the first key
                first_key = list(file.keys())[0]
                data = np.array(file[first_key])
                return data
        except Exception as e:
            print(f"Error reading with h5py: {e}")
            return None
    
    
    file_path = 'C:/Users/haloe/Documents/CodeWriting/pythonCodes/HanLab/InverseFunction/cbed_atomic_res.mat'
    
    """Read the 4D dataset as a numpy array from .mat file.
       the shape of the data is (x, y, kx, ky)."""
    # data = read_4D(file_path)
    
    atomic_res_data = read_mat_file(file_path)
    atomic_res_data = atomic_res_data['cbed']
    
    atomic_res_data = HyperData(atomic_res_data)
    atomic_res_data = atomic_res_data.swap_coordinates()
    atomic_res_data = atomic_res_data.alignment(r_mask=20)[0]
    
    
    #%%
    
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
        This function is based on MATLAB code written by Colin Ophus (MM/YYYY)."""
        nx, ny = size
        qx = np.fft.fftfreq(nx, spacing)
        qy = np.fft.fftfreq(ny, spacing)
        qxa, qya = np.meshgrid(qx, qy, indexing='ij')
        return qxa, qya
    
    
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
                if np.sign(np.sum(window)) != np.sign(center_val):
                    corrected_img[i, j] *= -1
    
        return corrected_img
    
    
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
    
    
    def fix_tilt_and_height(height_map):
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
        height_map = height_map - np.mean(height_map[0:25, 20:256])
    
        return height_map
    
    
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
    
    
    def iterative_reconstruction(xGrad, yGrad, iterations=10, threshold_percent=5, 
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
            hmap_fixed = fix_tilt_and_height(h_map)
    
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
    
    
    
    def getStrains(spots_order, centers_data, ref_centers=None, ang=49, gs=1, plot=True, order=False):
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
        
        print("g1: ", np.mean(g1,axis=(0,1)))
        print("g2: ", np.mean(g2,axis=(0,1)))
        
        """ Reference g's """
        ref_g1 = mean_pos[gs-1] - mean_pos[gs+2]
        ref_g2 = (mean_pos[gs] + mean_pos[gs+1]) / 2 - \
            (mean_pos[(gs+3) % 6] + mean_pos[(gs+4) % 6]) / 2
    
        """ G array """
        G_ref = np.array([[ref_g2[0], ref_g1[0]],
                          [ref_g2[1], ref_g1[1]]])
            
        """ R matrices """
        ang = ang
        R1 = np.array([[np.cos(ang/180*np.pi), np.sin(ang/180*np.pi)],
                      [-np.sin(ang/180*np.pi), np.cos(ang/180*np.pi)]])
    
        
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
        peak_idx, _ = find_peaks(counts)
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
    
    
    def getGradients(sol_array, rot_angle=0):
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
    
    def getPhiMapFromGradients(Grad, component, thetaMap):
        
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
    
    def remove_circular_region(image, centers, radius=6):
        """
        This function removes circular regions from a 2D array.
        """
    
        for center in centers:
            y, x = center
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if np.sqrt((i - y)**2 + (j - x)**2) <= radius:
                        image[i, j] = 0
        return image
    
    def predict_values(image, centers, radius=6, center_y=64, center_x=64,
                       n_estimators=100,learning_rate=0.1, max_depth=10):
        """
        Use XGBoost Regressor to interpolate gap regions of 2D array
        """
        # Apply the mask to the image
        if isinstance(radius, np.ndarray):    
            masked_image = image.copy()
            for idx, center in enumerate(centers):
                masked_image = remove_circular_region(masked_image, np.array([centers[idx]]), radius[idx])
        else:
            masked_image = remove_circular_region(image.copy(), centers, radius)
    
        # Create training data from the unmasked pixels using polar coordinates
        y, x = np.where(masked_image > 0)  
        r, theta = y,x
        # r, theta = cartesian_to_polar(y, x, center_y, center_x)
        X_train = np.column_stack((r, theta))
        y_train = masked_image[y, x]
    
        # Initialize and fit the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=n_estimators, 
                                 learning_rate=learning_rate, max_depth=max_depth)
        model.fit(X_train, y_train)
    
        # Predict the values of the masked pixels
        y_test, x_test = np.where(masked_image == 0)
        r_test, theta_test = y_test, x_test
        # r_test, theta_test = cartesian_to_polar(y_test, x_test, center_y, center_x)
        X_test = np.column_stack((r_test, theta_test))
        predictions = model.predict(X_test)
    
        # Fill in the predictions in the image
        predicted_background = masked_image.copy()
        predicted_background[y_test, x_test] = predictions
    
        return predicted_background
    
    
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
    
    def max_dp(data):
        """
        Calculate a 2D diffraction of the overall maximum int per pixel.
    
        Parameters:
        data (numpy.ndarray): 4D STEM dataset.
    
        Returns:
        numpy.ndarray: 2D mask of standard deviations with same shape as diffraction pattern.
        """
        # Validate the shape of the data
        if len(data.shape) != 4:
            raise ValueError("Data must be a 4D array with shape (A, B, C, C)")
    
        # Calculate the standard deviation for each pixel across all diffraction patterns
        max_dp = np.max(data, axis=(0, 1))
    
        return max_dp
    
    
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
    
    def plot_centers_of_mass_with_histograms(centers_data, colors, labels=None, drawConvexHull=True,
                                             transparency=0.5, hist_height=0.16, bins=150,
                                             alpha=0.5, density=True,):
        """
        Plots the centers of mass for all features from each dataset, centered at (0, 0),
        with histograms of the x and y coordinates.
    
        centers_data: The dataset with shape (n_datasets, A1, A2, n_spots, 2)
        hist_height: Height of the histograms as a fraction of total figure height
        """
        
        # Check data compatibility
        assert type(drawConvexHull) is bool, "'drawConvexHull' must be a boolean (True/False) variable"
        assert type(density) is bool, "'density' must be a boolean  (True/False) variable"
        assert len(colors) == centers_data.shape[0], "The number of colors must match the number of datasets to plot."
        assert all(isinstance(item, str) for item in colors), "Not all elements are strings."
    
        n_datasets = centers_data.shape[0]
    
        # Create the main plot
        fig = plt.figure(figsize=(10, 10))  
        ax_scatter = plt.axes([0.1, 0.1, 0.65, 0.65])
        ax_histx = plt.axes([0.1, 0.75, 0.65, hist_height], sharex=ax_scatter)
        ax_histy = plt.axes([0.75, 0.1, hist_height, 0.65], sharey=ax_scatter)
    
        # Disable labels on histogram to prevent overlap
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)
    
        # Initialize standard deviation lists
        std_dev_y = np.zeros(n_datasets)
        std_dev_x = np.zeros(n_datasets)
    
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
    
                # Scatter plot for each feature of the dataset
                ax_scatter.scatter(x_coords - x_mean, y_coords - y_mean, color=colors[i], alpha=transparency)
                
            # Calculate and store standard deviations
            std_dev_y.append(np.std(all_y_coords))
            std_dev_x.append(np.std(all_x_coords))
                
            # Combine all x and y coordinates
            combined_coords = np.column_stack((all_x_coords, all_y_coords))
    
            # Draw convex hull for the combined coordinates of the dataset
            if drawConvexHull and len(combined_coords) > 2:
                hull = ConvexHull(combined_coords)
                for simplex in hull.simplices:
                    ax_scatter.plot(combined_coords[simplex, 0], combined_coords[simplex, 1], color=colors[i], linewidth=2)
    
            # Add label for the dataset
            if labels is not None:
                ax_scatter.plot([], [], color=colors[i], label=labels[i])
            else:
                ax_scatter.plot([], [], color=colors[i], label=f'Dataset {i+1}')
    
            # Plot histograms
            ax_histx.hist(all_x_coords, bins=bins, color=colors[i], alpha=alpha, 
                          density=density, label=rf'$\sigma$ = {std_dev_x[-1]:.2f}')
            ax_histy.hist(all_y_coords, bins=bins, color=colors[i], alpha=alpha, orientation='horizontal', 
                          density=density, label=rf'$\sigma$ = {std_dev_y[-1]:.2f}')
    
    
        # Set labels and title for the scatter plot
        ax_scatter.set_xlabel('kx')
        ax_scatter.set_ylabel('ky')
        ax_scatter.set_title('Centers of Mass with Histograms')
        ax_scatter.legend()
    
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
    
    def get_clusters(dataset, n_PCAcomponents, n_clusters, r_centerBeam, std_Threshold = 0.2,
                     plotStdMask=True, plotScree=True, plotClusterMap=True, plot3dClusterMap=False, 
                     filter_size=3, cluster_cmap=None, filter_iterations=1,outer_ring=None, polar=False):
        
        """
        Function that returns cluster dataset 
        """
        
        assert len(dataset.shape)==4, "'dataset' must be 4-dimensional"
        
        # Remove center beam
        A,B,C,D = dataset.shape
        
        if not polar:
            dataset_noCenter = remove_center_beam(dataset, r_centerBeam, outer_ring=outer_ring)
        
        else:
            dataset_noCenter = np.zeros_like(dataset)
            if outer_ring is not None:
                dataset_noCenter[:,:,r_centerBeam:outer_ring] = dataset[:,:,r_centerBeam:outer_ring]
            else:
                dataset_noCenter[:,:,r_centerBeam:] = dataset[:,:,r_centerBeam:]
        
        # Find pixels of high variation
        dataset_stdev = get_kspace_stdDev(dataset_noCenter)
            
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
        data_reduced = pca.fit_transform(np.log(rem_negs(dataset_noCenter)))
        
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
    
    def getSNR(dataset, signalMask, resBgMask, noiseMask, returnArray=False):
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
            
    def getNoise(dataset, noiseMask):
        
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
    
    def remove_bg(dp, min_sigma=0.85, 
                      max_sigma=2.5, 
                      num_sigma=100, 
                      threshold=85,
                      bg_frac=0.99,
                      mask_radial_frac=0.9,
                      min_distance_factor=0.8,
                      max_distance_factor=1.5,
                      search_radius_factor=2.5, # When refining centers of mass
                      iterations=5,
                      n_estimators = 60, learning_rate = 0.1, max_depth = 50, # Predictor model parameters
                      radii_multiplier=1.5, radii_slope=1.38,
                      vmin=5,vmax=16, get_bg = True,
                      plotAxes=False,plotInput=False,plotCenters=True,plotBg=False,plotResult=True,):
        
        A, B = dp.shape
        mask = make_mask((A//2,B//2), r=(A+B)//4*mask_radial_frac) # Define region where spots are to be found
        
        # Extract centers and radii automatically
        blobs_log = feature.blob_log(dp*mask, min_sigma=min_sigma, 
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
        min_threshold = np.mean(np.min(distances,axis=0))*min_distance_factor  # Minimum acceptable distance to the nearest neighbor.
        max_threshold = np.mean(np.min(distances,axis=0))*max_distance_factor  # Maximum acceptable distance to the nearest neighbor.
    
        # Filter blobs. Keep a blob if its nearest neighbor is neither too close nor too far.
        filtered_indices = np.where((nearest_neighbor_dist >= min_threshold) 
                                    & (nearest_neighbor_dist <= max_threshold)
                                    & (((blobs_array[:,0]-A//2)**2+(blobs_array[:,1]-B//2)**2)**.5 <= (A+B)//4*mask_radial_frac*(8/9)))[0]
    
        # Apply filter
        blobs_log = np.zeros((len(filtered_indices), 3))
        blobs_log[:, :2] = blobs_array[filtered_indices]
        blobs_log[:, 2] = r_vals[filtered_indices]
        
        
        if plotCenters:
        
            plt.figure(figsize=(10, 10))
            plt.imshow(np.log(dp), cmap='turbo',vmin=5,vmax=16)
            plt.axis('off')
        
            # Circle around each blob
            for blob in blobs_log:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                plt.gca().add_patch(c)
        
            plt.show()
        
        # Refine peak centers using CoM
        peak_centers = getCenters(dp,r=blobs_log[:,2]*search_radius_factor,
                                  coords=blobs_log[:,:2], iterations=iterations)
        
        distance_to_center = np.linalg.norm(peak_centers-(A+B)//4,axis=1)
        radii = blobs_log[:,2]*radii_multiplier + radii_slope*(1 - distance_to_center/np.max(distance_to_center))
        
        if get_bg:
            # Masks sizes are based on computed radius of spot and distance to center (center spots with greater radius than far spots)
            dp_bg = predict_values(dp,centers=peak_centers,radius=radii,
                                   n_estimators=n_estimators,learning_rate=learning_rate, max_depth=max_depth)
    
        if plotInput:
            plotDP(dp, vmin=vmin,vmax=vmax,plotAxes=plotAxes)
        
        if plotBg and get_bg:
            plotDP(dp_bg, vmin=vmin,vmax=vmax,plotAxes=plotAxes)
    
        if plotResult:
            plotDP(rem_negs(dp-bg_frac*dp_bg), vmin=vmin,vmax=vmax,plotAxes=plotAxes,
                   title=f'ne={n_estimators},learning_rate={learning_rate}, max_depth={max_depth}')
        
        if get_bg:
            return peak_centers, radii, dp_bg
        else:
            return peak_centers, radii
    
    
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
    
        return rem_negs(polar_diff.T)
    
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
    
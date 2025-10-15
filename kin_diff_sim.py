"""
Generate theta (tilt), phi (tilt axis), and intensity data based on kinematic 
diffraciton model. 

Based on MATLAB Code by Suk Hyun Sung (See reference below)
[Reference]

Python script by Adan J. Mireles
  Rice University
  February 2023
  Updated August 2024 for generalizability
"""
# Import packages
import os
import re

# Laptop directory
# os.chdir('C:/Users\haloe/Documents/CodeWriting/pythonCodes/HanLab/InverseFunction')
# Lambda directory
# os.chdir('/home/han/Users/adan/pythonCodes/InverseFunction')

import numpy as np
from time import perf_counter

# =============================================================================
# Helper Functions/Definitions
# =============================================================================

def eDiff_Wavenumber(kev):
    wav = electronwavelength(kev)
    k = 2*np.pi / wav
    return k

def electronwavelength(kev):
    wav = 12.3986/np.sqrt((2*511.0 + kev)*kev)
    return wav

def f_element(q, element):
    """
    Compute the scattering factor F(q) for a given element using a sum of
    Lorentzian and Gaussian terms. The formula is:

        F(q) = sum over i of [ a[i] / (q^2 + b[i]) + c[i] * exp(-d[i] * q^2 ) ],

    where q^2 = (q / (2*pi))^2. The parameters a, b, c, d differ by element.    
    
    Based on parameterization and values from "Advanced Computing in Electron
    Microscopy" by Dr. Earl J. Kirkland
    
    By Robert Hovden [MATLAB]
    Oct 6, 2011
    
    Modified by Suk Hyun Sung [MATLAB]
    Aug 20, 2022
    
    Translated function to Python and modified by Adan Mireles to generalize 
    for an arbitrary element in TMD.
    February 2023
    """
    
    if element == 'Mo' or element == 42:
        a = np.array([6.10160120e-001, 1.26544000e+000, 1.97428762e+000])
        b = np.array([9.11628054e-002, 5.06776025e-001, 5.89590381e+000])
        c = np.array([6.48028962e-001, 2.60380817e-003, 1.13887493e-001])
        d = np.array([1.46634108e+000, 7.84336311e-003, 1.55114340e-001])
    
    elif element == 'Se' or element == 34:
        a = np.array([9.58390681e-001, 6.03851342e-001, 1.90828931e+000])
        b = np.array([1.83775557e-001, 1.96819224e+002, 2.15082053e+000])
        c = np.array([1.73885956e-001, 9.35265145e-001, 8.62254658e-003])
        d = np.array([3.00006024e-001, 4.92471215e+000, 2.12308108e-002])
        
    elif element == 'W' or element == 74:
        a = np.array([9.24910685e-001, 2.75554570e+000, 3.30440006e+000])
        b = np.array([1.28633770e-001, 7.65826749e-001, 1.34471170e+001])
        c = np.array([3.29973862e-001, 1.09916444e+000, 2.06498883e+002])
        d = np.array([1.98218895e-001, 1.35087534e+001, 3.38918459e+002])
    
    elif element == 'Te' or element == 52:
        a = np.array([2.09383882, 1.56949519, 1.30941993])
        b = np.array([12.6856869, 1.21236537, 0.166633292])
        c = np.array([0.0698067804, 1.04969537, 0.555592435])
        d = np.array([0.0830817576, 7.43147857, 0.617487676])
    
    F = np.zeros(np.shape(q))
    q2 = (q/(2*np.pi))**2
    
    for i in range(3): #three parameters
        F = F + a[i]/(q2 + b[i]) + c[i]*np.exp(-d[i]*q2)
                      
    return F

def eDiff_ScatteringVector(k, qx, qy, phi, theta):
    #The Ewald Sphere has an origin that changes with tilt and rotation
    qzo = k*np.cos(theta)
    qxo = k*np.sin(theta)*np.cos(phi)
    qyo = k*np.sin(theta)*np.sin(phi) 

    # With Ewald Sphere origin and qx, qy components, we can determine qz
    qz  = qzo - np.sqrt(k**2 - (qx-qxo)**2 - (qy-qyo)**2)
    qz = qz[..., None] # Reshape to single column
 
    N = len(qz)
    
    qxy = np.array([qx, qy])
    qxy_tile = np.tile(qxy, (N,1))     # Copy qx and qy N times
    
    # q = np.hstack([qxy_tile, qz])      # Append qz values to (qx, qy) array  
    q = np.column_stack((qxy_tile, np.asarray(qz).reshape(-1,1)))
    return q

def BraggRod(h, k, keV, phi, theta, a, material='MoSe2',):
    
    K0 = eDiff_Wavenumber( keV )
    b = 4*np.pi/(np.sqrt(3)*a)
    L =  a/2      # in Ang
    b1 = b*np.array([0,1])
    b2 = b*np.array([-np.sqrt(3)/2, 1/2])
       
    qx = h*b1[0] + k*b2[0]
    qy = h*b1[1] + k*b2[1]
    
    q = eDiff_ScatteringVector(K0, qx, qy, phi, theta) 

    kz = q[:,2]

    K = np.sqrt(3*b**2 + kz**2)
    
    species, subscripts = get_chemical_composition(material)
    
    if len(species) == 2:
        Int = np.real((f_element(K, species[0]) + \
                       subscripts[1]*f_element(K, species[1])*np.cos(kz*L))**2)
    
    return q, Int


def _decompose_chemical_formula(formula):
    """
    Decomposes a chemical formula into its constituent elements and their quantities.
    
    This function parses a chemical formula string and returns a dictionary where the keys
    are element symbols and the values are their respective quantities as integers. If an
    element is mentioned without a quantity, it defaults to 1.
    
    Parameters
    ----------
    formula : str
        The chemical formula string to be decomposed.
    
    Returns
    -------
    elements : dict
        A dictionary containing elements and their corresponding quantities.
    
    Examples
    --------
    >>> _decompose_chemical_formula('H2O')
    {'H': 2, 'O': 1}
    
    >>> _decompose_chemical_formula('C6H12O6')
    {'C': 6, 'H': 12, 'O': 6}
    
    Notes
    -----
    This function uses regular expressions to parse the formula. It supports elements with
    multi-letter symbols and quantifies immediately following the element symbol.
    """
    
    if not type(formula)  == str:
        raise ValueError("The input 'formula' must be a string in the format of an empirical formula")
    
    # This pattern matches capital letters possibly followed by lower case letters (element symbols)
    # and then optionally followed by numbers (quantities)
    pattern = r'([A-Z][a-z]*)(\d*)'
    
    components = re.findall(pattern, formula)
    
    elements = {}  # Dictionary to hold element symbols and their quantities
    
    for element, quantity in components:
        if quantity == '':
            quantity = 1  # If no quantity specified, default to 1
        else:
            quantity = int(quantity)  # Convert quantity to integer
        
        if element in elements:
            elements[element] += quantity  # Sum quantities for the same element
        else:
            elements[element] = quantity
    
    return elements

def get_chemical_composition(material):
    """
    Extracts the chemical composition from a formula string.
    
    This function analyzes a chemical formula string to identify the constituent 
    elements and their respective counts. It uses the `_decompose_chemical_formula` 
    function to parse the formula and returns the elements and their subscripts 
    in separate lists.
    
    Parameters
    ----------
    material : str
        A string representing the chemical formula of the material.
    
    Returns
    -------
    species : list of str
        A list of the chemical elements present in the material.
    subscripts : list of int
        A list of the corresponding subscripts for the elements in the material, 
        aligned with the `species` list.
    
    Examples
    --------
    >>> get_chemical_composition('H2O')
    (['H', 'O'], [2, 1])
    
    Notes
    -----
    The function relies on correct chemical notation being used in the input 
    formula where each element is represented by its symbol starting with a 
    capital letter possibly followed by a lowercase letter, followed by an 
    integer subscript.
    """
    
    chem_composition = _decompose_chemical_formula(material)
    species = list(chem_composition.keys())
    subscripts = [chem_composition[key] for key in species]
    
    return species, subscripts

#%%
# =============================================================================
# Run Through: Main Code to Run
# =============================================================================

# Folder in which to save data
data_path = 'generatedData'

"""
Define the reciprocal lattice coordinates of Bragg peaks
"""
hks = np.array([
                # 1st order
                [[ 1,  0],
                [ 0,  1],
                [-1,  1],
                [-1,  0],
                [ 0, -1],
                [ 1, -1]],
                # 2nd order
                [[ 1,  1],
                [-1,  2],
                [-2,  1],
                [-1, -1],
                [ 1, -2],
                [ 2, -1]],
                # # 3rd order
                [[ 2,  0],
                [ 0,  2],
                [-2,  2],
                [-2,  0],
                [ 0, -2],
                [ 2, -2]]
               ])


def compute_intensity(phi, theta, h, k, 
          keV=80, material='MoSe2', a=3.32):
    """
    Simplified version of the function
    Input is in radians
    """
    
    if not isinstance(theta, np.ndarray):
        theta = np.atleast_1d(np.array([theta]))
    
    qs, Int = BraggRod(h, k, keV, phi, theta, material=material, a=a)    
    
    return Int


def compute_intensities(phi, theta, order=2, hks=hks, 
          keV=80, normalization='flat_max',
          material='MoSe2',  a=3.32, conjugate=False):
    """
    Compute the six spot intensities at specific tilt and direction for a given 
    material and diffraction order.

    Parameters
    ----------
    phi : float
        Tilt direction angle in radians.
    theta : float
        Tilt angle in the plane of the sample in radians.
    order : int
        Diffraction order (1, 2, or 3).
    hks : array
        Array containing the Miller indices for diffraction spots.
    keV : float, optional
        Accelerating voltage in keV (default is 80 keV).
    normalization : float or str, optional
        Normalization factor for intensity calculations, can be 'max' to normalize 
        by the maximum intensity in the six spots or 'flat_max' (default) to 
        normalize based on maximum possible intensity of 86.43649588.
    material : str, optional
        Material type (default is 'MoSe2').
    a : float, optional
        Lattice parameter (default is 3.32, for MoSe2 TMD).
    conjugate : bool
        If true, computes the conjugate intensities (Default is False).

    Returns
    -------
    array
        Array of optionally normalized spot intensities (or conjugate 
        intensities if 'conjugate' is True)

    Raises
    ------
    ValueError
        If 'order' is not 1, 2, or 3.

    Example
    --------
    # We calculate intensities for 2nd order spots at (phi, theta) = (110, 10) # degrees
    >>> compute_intensities(phi=np.radians(110), theta=np.radians(10), 
                            order=2, normalization='flat_max')
    Out:
        array([0.49446022, 0.94047028, 0.65394333, 
               0.37460648, 0.86848257, 0.76673135]) 
    """
    
    if normalization == 'max':
        norm = 1
    elif normalization == 'flat_max':
        norm = 86.43649588
    else:
        norm = normalization
    
    if order not in [1,2,3]:
        raise ValueError("Input parameter 'order' should be an integer equal to 1, 2, or 3.") 
        
    num_peaks = len(hks[order-1])
    
    all_ints = np.zeros((num_peaks, len(theta)) if isinstance(theta, (list, np.ndarray)) and len(theta) > 1 else num_peaks)

    for i in range(num_peaks):
        all_ints[i] = compute_intensity(phi, theta, 
                            hks[order-1, i,0], hks[order-1,i,1], 
                            keV=keV, material=material, a=a)
        
    if conjugate:
        
        # Initialize array of conjugate intensities 
        conj_ints = np.zeros(3)
        
        for i in range(3):
            conj_ints[i] = (all_ints[i] + all_ints[i+3])/2
        
        all_ints = conj_ints

    if normalization == 'max':
        all_ints /= np.max(all_ints)
    else:
        all_ints /= norm
    
    return all_ints

#%%

from sklearn.metrics.pairwise import cosine_similarity

def get_tilt(intensities, method='grid_search', loss='euclidean_distance', 
              simDataArray=None, domain1=None, domain2=None, 
              initial_guess1=None, initial_guess2=None, **sim_kwargs):
    """
    Find the tilt axis and surface tilt that best match the provided set of 
    intensities based on a library of possible solutions.
    """
    
    if loss not in ['cosine_similarity', 'euclidean_distance', 'manhattan_distance']:
        raise ValueError(""""Input parameter 'loss' is not valid or implemented. \n
                             Valid inputs are: 'cosine_similarity', 'euclidean_distance', 
                            'manhattan_distance'""") 
    
    if method == 'grid_search':
            
        # Extract the data portion without phi, theta columns
        sim_intensities = simDataArray[:, 2:]
    
        if loss == 'cosine_similarity':
        
            # Compute the Cosine Similarity
            similarities = cosine_similarity(sim_intensities, [intensities])
            sol_idx = np.argmax(similarities)
        
        elif loss == 'euclidean_distance':
        
            # Compute the squared Euclidean distance
            # distances = np.sqrt(np.sum((sim_intensities - intensities)**2, axis=1))
            distances = np.linalg.norm(sim_intensities - intensities, axis=1)
            sol_idx = np.argmin(distances)
        
        elif loss == 'manhattan_distance':
            
            # Compute the Manhattan distance
            distances = np.sum(np.abs(sim_intensities - intensities), axis=1)
            sol_idx = np.argmin(distances)
                    
        phi_sol, theta_sol = simDataArray[sol_idx][:2]
    
    elif method == 'bayesian':
        
        phi_sol, theta_sol = find_optimal_params(intensities, domain1, domain2, criterion=loss, 
                                                 initial_guess1=None, initial_guess2=None, **sim_kwargs)

    return phi_sol, theta_sol

def get_tilts(intensity_array, show_result=True, figsize=(12, 5), fraction=0.046, **get_tilt_args):
    """
    Fing the optimal tilt descriptors for a full two-dimensional map.
    """
    
    Ny, Nx, _ = intensity_array.shape
    tilts = np.zeros((Ny, Nx, 2))

    for i, row in tqdm(enumerate(intensity_array), total=Ny, desc="Finding optimal (phi, theta)"):
        for j, ints in enumerate(row):
            
            tilts[i,j] = get_tilt(intensities=ints, **get_tilt_args)
    
    if show_result:
        
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # First subplot
        im1 = axs[0].imshow(np.degrees(tilts[:, :, 0]), cmap='hsv')
        axs[0].axis('off')
        axs[0].set_title('Azimuthal\nAngle', fontsize=8)
        cbar1 = fig.colorbar(im1, ax=axs[0], fraction=fraction)
        cbar1.ax.set_title(r'$\phi$ (°)', fontsize=8, pad=5)
        
        # Second subplot
        im2 = axs[1].imshow(np.degrees(tilts[:, :, 1]), cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Elevation\nAngle', fontsize=8)
        cbar2 = fig.colorbar(im2, ax=axs[1], fraction=fraction)
        cbar2.ax.set_title(r'$\theta$ (°)', fontsize=8, pad=5)
        
        # Adjust layout to make the subplots fit well
        plt.tight_layout()
        plt.show()
    
    return tilts

#%%

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

min_theta, max_theta = (0, 40)

# Define the domains with named dimensions
domain1 = [Real(0, np.pi, name='phi'), Real(np.radians(min_theta), np.radians(max_theta), name='theta')]
domain2 = [Real(np.pi, 2*np.pi, name='phi'), Real(np.radians(min_theta), np.radians(max_theta), name='theta')]

# Define objective function
def objective(params, target, criterion='mse', **function_kwargs):
    phi, theta = params
    sim_vector = compute_intensities(phi, theta, **function_kwargs)
    
    if criterion == 'cosine_similarity':
        return -cosine_similarity([sim_vector], [target])[0][0]
    elif criterion == 'modified_cosine':
        return -cosine_similarity([sim_vector], [target])[0][0]**60
    elif criterion == 'mse':
        return mean_squared_error(sim_vector, target)
    elif criterion == 'euclidean_distance':
        return euclidean(sim_vector, target)
    else:
        raise ValueError("Unknown criterion: " + criterion)

# Optimization function
def optimize_target(target, criterion, domain, initial_guess=None, **function_kwargs):
    # if initial_guess is not None:
    #     validate_initial_guess(initial_guess, domain)
    
    @use_named_args(domain)
    def wrapped_objective(**params):
        return objective((params['phi'], params['theta']), target, criterion, **function_kwargs)
    
    return gp_minimize(wrapped_objective, dimensions=domain, x0=initial_guess, n_calls=2, random_state=None, 
                       n_initial_points=1, verbose=False, n_points=25, acq_func='gp_hedge', acq_optimizer='auto', n_jobs=-1, noise=1e-10)

# def validate_initial_guess(initial_guess, domain):
#     for i, val in enumerate(initial_guess):
#         lower_bound, upper_bound = domain[i].bounds
#         if not (lower_bound <= val <= upper_bound):
#             raise ValueError(f"Initial guess {val} for parameter {i} is out of bounds: [{lower_bound}, {upper_bound}]")


# Main function to process input and find optimal (phi, theta)
def find_optimal_params(input_data, domain1, domain2=None, criterion='mse', initial_guess1=None, initial_guess2=None, **function_kwargs, ):
    # Ensure the input data is 3D
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, 1, -1)
    elif input_data.ndim == 2:
        input_data = input_data.reshape(-1, 1, input_data.shape[1])

    A, B, _ = input_data.shape
    optimal_params = np.zeros((A, B, 2))
    
    for a in range(A):
        for b in range(B):
            target = input_data[a, b]
            result1 = optimize_target(target, criterion, domain1, initial_guess=initial_guess1, **function_kwargs)
            
            if domain2 is not None:
                result2 = optimize_target(target, criterion, domain2, initial_guess=initial_guess2, **function_kwargs)
            
                # Choose the best result between the two domains
                if result1.fun < result2.fun:
                    optimal_params[a, b] = result1.x
                else:
                    optimal_params[a, b] = result2.x
            
            else:
                optimal_params[a, b] = result1.x
            
    return np.squeeze(optimal_params)

#%%

if __name__ == '__main__':
    phi_test, theta_test = np.random.random(2)
    phi_input = phi_test*2*np.pi
    theta_input = theta_test*np.radians(max_theta-min_theta) + np.radians(min_theta)

    data = compute_intensities(phi_input, theta_input)
    
    criterion = 'euclidean_distance'  # Choose the criterion ('cosine_similarity', 'mse', 'rmse', 'euclidean_distance')
    
    phi1 = phi_input
    if 0 < phi_input < np.pi:
        phi1 = phi_input 
        phi2 = phi1 + np.pi
    else:
        phi1 = phi_input - np.pi 
        phi2 = phi_test
    
    domain1 = [Real(phi1 - np.radians(10), phi1 + np.radians(10), name='phi'), Real(theta_input - np.radians(1), theta_input + np.radians(1), name='theta')]
    domain2 = [Real(phi2 - np.radians(10), phi2 + np.radians(10), name='phi'), Real(theta_input - np.radians(3), theta_input + np.radians(3), name='theta')]
        
    optimal_parameters = find_optimal_params(data, domain1, domain2=None, criterion=criterion, 
                                              initial_guess1=[phi1*(1+np.random.random()*0.05), theta_input*(1+np.random.random()*0.05)], 
                                              initial_guess2=[phi2*(1+np.random.random()*0.05), theta_input*(1+np.random.random()*0.05)], 
                                             order=2, normalization='flat_max', material='MoSe2', a=3.32)
    
    print("Tested values where:", (np.degrees(phi_input), np.degrees(theta_input)))
    print("Output values where:", np.degrees(optimal_parameters))


#%%

import matplotlib.pyplot as plt

def optimize_and_plot(phi_test, theta_test, criteria, initial_guess1=None, initial_guess2=None, **function_kwargs):
    fig, axes = plt.subplots(1, len(criteria), figsize=(20, 5))

    for i, criterion in enumerate(criteria):
        theta_input = []
        theta_output = []
        phi_values = []
        
        for phi, theta in zip(phi_test, theta_test):
            data = compute_intensities(phi, theta, **function_kwargs)
            optimal_params = find_optimal_params(data, criterion, initial_guess1, initial_guess2, **function_kwargs)
            optimal_phi, optimal_theta = optimal_params[0, 0]  # Assuming a single (phi, theta) pair output
            
            theta_input.append(np.degrees(theta))
            theta_output.append(np.degrees(optimal_theta))
            phi_values.append(optimal_phi)
        
        scatter = axes[i].scatter(theta_input, theta_output, c=np.degrees(np.array(phi_values)), cmap='viridis')
        axes[i].set_title(criterion)
        axes[i].set_xlabel('Theta Input (degrees)')
        axes[i].set_ylabel('Theta Output (degrees)')
        cbar = fig.colorbar(scatter, ax=axes[i])
        cbar.set_label('Phi Value (degrees)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    criteria = ['modified_cosine', 'cosine_similarity', 'mse', 'euclidean_distance']
    
    # Define the domain for phi and theta
    min_theta = 0
    max_theta = 40
    num_samples = 50
    
    # Generate random test pairs
    phi_test = np.random.uniform(0, 2*np.pi, num_samples)
    theta_test = np.random.uniform(np.radians(min_theta), np.radians(max_theta), num_samples)
    
    optimize_and_plot(phi_test, theta_test, criteria, order=2, normalization='flat_max', material='MoSe2', a=3.32)


#%% Generate data (all desired spots) --> EFFICIENT

from itertools import product
from tqdm import tqdm

def generate_library(phi_range, num_phis, 
                     theta_range, num_thetas, 
                     save=False, fname=None, **f_kwargs):
    """
    Generate a library of kinematic diffraction simulations for a grid of 
    phi (tilt direction or azimuthal angle) and theta (tilt or elevation angle) values.

    Parameters
    ----------
    phi_range : tuple
        A tuple (min_phi, max_phi) specifying the range of phi values in degrees.
    num_phis : int
        The number of phi values to sample within the specified range.
    theta_range : tuple
        A tuple (min_theta, max_theta) specifying the range of theta values in degrees.
    num_thetas : int
        The number of theta values to sample within the specified range.
    save : bool, optional
        If True, save the generated data array to a .npy file. Default is False.
    fname : str, optional
        The filename for saving the data array. If None and save is True, the 
        filename will be generated automatically based on theta_range. Default is None.
    **f_kwargs : dict
        Additional keyword arguments to pass to the compute_intensities function.

    Returns
    -------
    main_data_array : numpy.ndarray
        An array of shape (num_phis * num_thetas, 8) containing the generated 
        phi, theta, and simulated intensities.

    Notes
    -----
    The first two columns of the returned array correspond to phi and theta 
    values in radians. The remaining columns contain the simulated intensities.
    The compute_intensities function is assumed to generate a 6-element array 
    of intensities for each combination of phi and theta.
    If save is True and fname is not provided, the data array will be saved as 
    'simInts_thetaMax_{max_theta}deg.npy'.
    """
    
    #TODO: add the option to specify delta_phi and delta_theta
    
    phis = np.radians(np.linspace(phi_range[0], phi_range[1], int(num_phis)).T) 
    thetas = np.radians(np.linspace(theta_range[0], theta_range[1], int(num_thetas)).T) 
    
    num_phth = len(phis) * len(thetas)
    main_data_array = np.zeros((num_phth, 8))     
        

    # Iterate directly over the product generator
    for idx, (phi, theta) in tqdm(enumerate(product(phis, thetas)), 
                                  total=num_phth, desc=r'Generating library of (phi, theta, intensities)'):
        
        main_data_array[idx][0] = phi
        main_data_array[idx][1] = theta
        
        sim_ints = compute_intensities(phi, theta, **f_kwargs)
        main_data_array[idx][2:8] = sim_ints

    if save:
        if fname is None:
            fname = f'simInts_thetaMax_{int(theta_range[1])}deg.npy'
            np.save(fname, main_data_array)
        else:
            if not fname.endswith('.npy'):
                fname += '.npy'            
            np.save(fname, main_data_array)
        print(f'Saved all data as: {fname}')
        
    return main_data_array
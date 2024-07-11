"""
abTEM helper functions.

Author: Adan J Mireles
Date: April 2024.
Last updated: June 2024.
"""

from abtem import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential. 
        Provide either "sampling" (spacing between consecutive grid points) or 
        "gpts" (total number of grid points).
    
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å]. Provide either "sampling" or "gpts".
"""

def abTEM2numpy(abTEM_dp_object, resize_dims=None):
    """
    Convert a DiffractionPatterns object from abTEM to a numpy array with a user-defined 
    """

    if abTEM_dp_object.is_lazy:
        abTEM_dp_object.compute()
        
    if resize_dims:
        array = cv2.resize(abTEM_dp_object.array, resize_dims, interpolation=cv2.INTER_LINEAR)
    
    return array
    

def simulate_diff(atom_model, substrate=None, interlayer_dist=2, vacuum=2,
                  theta_tilt=0, phi_rot=0, rotZ=0, units='deg',
                  energy=80e3, semiangle_cutoff=0.5, sampling_potential=0.1, 
                  slice_thickness=0.5, sampling_probe=None, gpts=1000, device='cpu', 
                  max_angle=None, get_probe_info=False, show_model=False,
                  frozen_phonons=False, num_configs=10, sigmas=0.1, seed=100, ensemble_mean=True,
                  grid=None,auto_crop=True):
    """
    Simulate a diffraction pattern from an atomic model with options for rotation, tilting, and substrate integration.

    Parameters
    ----------
    atom_model : ASE.Atoms
        Atomic configuration of the material of interest.
    substrate : ASE.Atoms, optional
        Atomic configuration of the substrate.
    interlayer_dist : float, optional
        Distance between the atom model and the substrate in angstroms.
    vacuum : float, optional
        Amount of vacuum padding around the atom model's cell in angstroms.
    theta_tilt : float, optional
        Tilt angle around the phi_rot axis.
    phi_rot : float, optional
        Rotation angle of the axis about which theta_tilt is measured.
    rotZ : float, optional
        Additional rotation around the Z-axis.
    energy : float, optional
        Electron beam energy in electron volts.
    semiangle_cutoff : float, optional
        Semi-angle cutoff of the electron probe in milliradians.
    sampling_potential : float, optional
        Sampling interval for the electrostatic potential in angstroms.
    slice_thickness : float, optional
        Thickness of each potential slice in angstroms.
    sampling_probe : float, optional
        Sampling interval for the electron probe in angstroms.
    gpts : int, optional
        Number of grid points for the potential.
    device : str, optional
        Computation device to use ('cpu' or 'gpu').
    max_angle : float, optional
        Maximum scattering angle for the diffraction pattern simulation.
    units : str, optional
        Units for tilt and rotation angles ('deg' for degrees or 'rad' for radians).
    get_probe_info : bool, optional
        If True, returns a plot of the probe's profile and its FWHM.
    show_model : bool, optional
        If True, displays the final atomic model.
    frozen_phonons : bool, optional
        If True, applies frozen phonon configurations.
    num_configs : int, optional
        Number of frozen phonon configurations.
    sigmas : float, optional
        Standard deviation of atomic displacements for frozen phonons.
    seed : int, optional
        Seed for random number generation in frozen phonons.
    ensemble_mean : bool, optional
        If True, averages results from all frozen phonon configurations.
    grid : tuple, optional
        Specifies the start and end points for a grid scan in the simulation.
    auto_crop : bool, optional
        Automatically crops the diffraction pattern to sqrt(1/2) times the 
        maximum angle. This corresponds to cropping a circle to its inscribed square.

    Returns
    -------
    abtem.measurements.DiffractionPatterns
        Simulated convergent beam electron diffraction pattern.

    Notes
    -----
    Substrate integration does not involve rotation, only alignment based on 
    the specified interlayer distance and vacuum padding.
    """
    
    # Check whether the sample was tilted or rotated and apply changes to material model
    if theta_tilt != 0 or rotZ != 0:
        atom_model = rot_lattice(atom_model, theta_tilt, phi_rot, rotZ, units, show=False)
    
    # Check whether the material model has a substrate
    if substrate is not None:
                  
        # Adjust position of material of interest with respect to the substrate
        atom_model.set_cell(substrate.cell)
        atom_model.center()
        
        # We ensure that the lowermost atom in the material of interest is at most 'interlayer_dist' from the substrate
        atom_model.positions[:, 2] += substrate.cell[2,2] - np.min(atom_model.positions[:,2]) + interlayer_dist
        
        # We combine the substrate and the material of interest into a single atom model
        atom_model = atom_model + substrate
        atom_model.center(axis=2, vacuum=vacuum)        
    
    # Apply thermal vibrations
    if frozen_phonons:
        atom_model_static = atom_model.copy()
        atom_model = FrozenPhonons(atom_model, num_configs=num_configs, 
                                  sigmas=sigmas, seed=seed, ensemble_mean=True)    
           
    potential = Potential(atom_model, sampling=sampling_potential,
                          gpts=gpts, slice_thickness=slice_thickness,
                          parametrization='kirkland', device=device, periodic=True)
    
    probe = Probe(energy=energy, semiangle_cutoff=semiangle_cutoff,)
    
    if get_probe_info:
        probe.grid.match(potential)
        probe.profiles().show();
        print(f"Probe FWHM = {probe.profiles().width().compute()} Å")
    
    # 4D-STEM Data Simulation
    if grid is not None:
        grid_scan = GridScan(
                    potential=potential,
                    start=grid[0],
                    end=grid[1],
                    sampling=probe.aperture.nyquist_sampling,
                    fractional=True)
        
        detector = PixelatedDetector()
        measurements = probe.scan(potential, scan=grid_scan, detectors=detector)
        
        return measurements        

    # Single Diffraction Pattern Simulation
    else: 
        
        # Run multislice algorithm
        probe_exit_wave = probe.multislice(potential=potential,)
                
        # Crop out diffraction data without information and disregard 10% of the outermost data
        if auto_crop:
            max_angle = np.min(probe_exit_wave.diffraction_patterns().max_angles)*np.sqrt(1/2)*0.9
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns(max_angle=max_angle)
        elif max_angle is not None:
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns(max_angle=max_angle)
        else:
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns()
        
        if show_model:
            if frozen_phonons:
                show_atoms(atom_model_static, plane='xz', title='Side View of Final Model')
            else:
                show_atoms(atom_model, plane='xz', title='Side View of Final Model')
        
        if frozen_phonons:
            return cbed_diffraction_pattern.mean(0)
        else:
            return cbed_diffraction_pattern


def rot_lattice(atom_model, theta_tilt=0, phi_rot=0, rotZ=0, units='deg', show=True,):
    """
    Rotate the lattice of an atomic model about specified angles and axes.

    This function performs a rotation transformation on the atomic model's lattice,
    based on given tilt and rotational parameters. The rotation can be performed in
    degrees or radians and can include an optional rotation around the Z-axis.

    Parameters
    ----------
    atom_model : ase.atoms.Atoms object
        The atomic model to be transformed.
    theta_tilt : float, optional
        The tilt angle around the computed axis. Default is 0.
    phi_rot : float, optional
        The rotation angle around the z-axis to define the tilt axis. Default is 0.
    rotZ : float, optional
        Additional rotation around the Z-axis. Default is None.
    units : str, optional
        Units for angular measurements. Accepts 'deg' for degrees or 'rad' for radians. Default is 'deg'.
    plot : bool, optional
        If True, plot the top and side views of the rotated lattice. Default is True.

    Returns
    -------
    tilted_atom_model : ase.atoms.Atoms object
        A new atomic model that has been rotated according to the specified parameters.

    Example
    --------
    >>> atom_model = read("my_atom_model.cif")
    >>> tilted_model = rot_lattice(atom_model, theta_tilt=30, phi_rot=45, plot=False)
    >>> print(tilted_model.positions)  # Outputs rotated positions of the atomic model

    Notes
    -----
    The rotations are applied first about the phi axis and optionally about the Z-axis.
    Rotation angles need to be specified in either degrees or radians based on 
    the 'units' parameter and must be the same units for all inputs.
    """

    if units == 'deg':
        # Convert tilt and rotation to radians if necessary
        phi_rot = np.radians(phi_rot)  # Only convert phi_rot if units are in degrees

    elif units != 'rad':
        # Validate units and convert theta_tilt if not already in degrees
        theta_tilt = np.degrees(theta_tilt)
        
        if rotZ != 0:
            rotZ = np.degrees(rotZ)
        
        raise ValueError("units must be 'rad' or 'deg'.")

    # Create a copy of the model and apply rotations
    tilted_atom_model = atom_model.copy()
    tilted_atom_model.rotate(theta_tilt,
                             (np.cos(phi_rot), np.sin(phi_rot), 0),
                             center='COU',
                            )
        
    if rotZ != 0:
        tilted_atom_model.rotate(rotZ, 'z', 
                                 center='COU',
                                )

    # Optionally show the rotated model
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        show_atoms(tilted_atom_model, ax=ax1, title='Top view', show_cell=True)
        show_atoms(tilted_atom_model, ax=ax2, plane='xz', title='Side view',show_cell=True)

    return tilted_atom_model

def add_poisson_noise(array, counts):

    # Remove any negative numbers and normalize
    noisy_array = np.copy(array)
    noisy_array[noisy_array < 0] = 0
    noisy_array = noisy_array/np.max(noisy_array)
    
    np.random.poisson(noisy_array, size=counts)
    

def show_atoms_top_and_side(atom_model, 
                              figsize: tuple = (12,4), 
                              show_cell: bool = True, 
                              title1: str ='Top View', 
                              title2: str ='Side View',
                              ):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    show_atoms(atom_model, ax=ax1, plane='xy', title=title1,show_cell=False)
    show_atoms(atom_model, ax=ax2, plane='xz', title=title2)
    

def plotDP(data,
           log01=False,
           title='Simulated Diffraction Pattern',
           axes01=True,
           vmin=0.000005,
           vmax=0.0005,
           r=None,
           cmap='turbo'):
    """
    Visualize a desired diffraction pattern
    """

    diff_shape = data.shape

    # Visualize
    plt.figure(figsize=(10, 10))
    # -12 rotation yields aligned pattern
    #im1 = plt.imshow(rotate((left_midpoint),0)*mask,vmin = 0, vmax = 1850, cmap='turbo')
    
    if log01:
        
        if r is not None:
            im1 = plt.imshow(np.log(data[int(diff_shape[0]/2)-r:int(diff_shape[0]/2)+r,
                                         int(diff_shape[1]/2)-r:int(diff_shape[1]/2)+r]),
                             vmin=np.log(vmin),
                             vmax=np.log(vmax),
                             cmap='turbo')
            
        else:
            im1 = plt.imshow(np.log(data),
                             vmin=np.log(vmin),
                             vmax=np.log(vmax),
                             cmap='turbo')
    else:
        
        if r is not None:
            im1 = plt.imshow(data[int(diff_shape[0]/2)-r:int(diff_shape[0]/2)+r,
                                  int(diff_shape[1]/2)-r:int(diff_shape[1]/2)+r],
                             vmin=vmin,
                             vmax=vmax,
                             cmap=cmap)
        else:
            im1 = plt.imshow(data,
                             vmin=vmin,
                             vmax=vmax,
                             cmap=cmap)            
    if axes01:

        plt.axis('on')
        
        if r is not None:
            plt.xticks(np.arange(0, 2*r, int(r/10)))
            plt.yticks(np.arange(0, 2*r, int(r/10)))

        # plt.grid()
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


def simDP(atom_model,
          theta_tilt=0,
          phi_rot=0,
          plot=True,
          vmin=0.000005,
          vmax=0.0005,
          r=None,
          crop=False,
          resize_dim=None,
          gpts=1000,
          energy=80e3, 
          semiangle_cutoff=0.5, 
          sampling_potential=0.1,
          slice_thickness=1,
          sampling_probe=0.04,
          log01=False,
          cmap='turbo',
          device='cpu',
          max_angle=None,
          max_frequency=None,
          rotZ=None,
          title='Simulated Diffraction Pattern'):

    cbed_pattern = simulate_diff(atom_model, 
                                 theta_tilt=theta_tilt, phi_rot=phi_rot, 
                                 energy=energy, semiangle_cutoff=semiangle_cutoff, 
                                 sampling_potential=sampling_potential,slice_thickness=slice_thickness,
                                 sampling_probe=sampling_probe,gpts=gpts,device=device,
                                 max_angle=max_angle, max_frequency=max_frequency,rotZ=rotZ)
    
    # ky, kx = cbed_pattern.array.shape
    # (cbed_pattern**0.15).show()
    # print(f"CBED pattern dimensions: ({ky}, {kx})")
    
    ky, kx = cbed_pattern.array.shape
    # kmin = np.min([ky, kx])
    # kround = kmin - (kmin % 10)
    
    # if resize_dim is not None:
    #     diff_image = cv2.resize(cbed_pattern.array, resize_dim)
    #     diff_image = diff_image.T
    
    # else:
    diff_image = cbed_pattern.array.T

    if plot:

        plotDP(diff_image, log01=log01, vmin=vmin, vmax=vmax, r=r, cmap=cmap, title=title)
        # cbed_pattern.block_direct().show()
    if crop:

        diff_shape = diff_image.shape

        diff_image = diff_image[(diff_shape[0]//2)-r:(diff_shape[0]//2)+r,
                                (diff_shape[1]//2)-r:(diff_shape[1]//2)+r]

    return diff_image

def make_mask(center, r, mask_dim=(128,128)):
    """
    Create a mask at a defined radius from a point (center of spot)
    """
    
    mask = np.zeros(mask_dim)
    y, x = center
    for i in range(mask_dim[0]):
        for j in range(mask_dim[1]):
            if (i-y)**2 + (j-x)**2 <= r**2:
                mask[i][j] = 1

    return mask

def extract_spot_features(dp, min_sigma=0.85, 
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
    # mask = make_mask((A//2,B//2), r=(A+B)//4*mask_radial_frac, mask_dim=(A,B)) # Define region where spots are to be found
    
    # Extract centers and radii automatically
    # blobs_log = feature.blob_log(dp*mask, min_sigma=min_sigma, 
    blobs_log = feature.blob_log(dp, min_sigma=min_sigma, 
                                 max_sigma=max_sigma, 
                                 num_sigma=num_sigma, 
                                 threshold=threshold,overlap=0)
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
    
        
    # Refine peak centers using CoM
    peak_centers = getCenters(dp,r=blobs_log[:,2]*search_radius_factor,
                              coords=blobs_log[:,:2], iterations=iterations)
    
    if plotCenters:
    
        plt.figure(figsize=(10, 10))
        plt.imshow(np.log(dp), cmap='turbo',vmin=vmin,vmax=vmax)
        plt.axis('off')
    
        # Circle around each blob
        for idx, blob in enumerate(blobs_log):
            y, x, r = blob
            y , x = peak_centers[idx]
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            plt.gca().add_patch(c)
    
        plt.show()
    
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

def getCenters(dp_data, spots_order=None, r=6, coords=None, num_spots_per_order=6, 
               plotSpots=False, iterations=1):
    """
    Generate an array spot centers for any DP
    """
    
    assert len(dp_data.shape) == 2, "'dp_data' must be of 2-dimensional"
    
    if spots_order is not None:
        
        # Define spot indeces to extract
        spot_adder = (spots_order - 1)*num_spots_per_order
    
        order_centers = np.zeros((num_spots_per_order, 2))
        for j in range(num_spots_per_order):
            order_centers[j] = getSpotCenter(coords[j + spot_adder][0],
                                             coords[j + spot_adder][1],
                                             r, dp_data, plotSpots,iterations=iterations)
            
    else:
        
        order_centers = np.zeros((len(coords), 2))
        
        for j in range(len(coords)):
            order_centers[j] = getSpotCenter(coords[j][0],
                                             coords[j][1],
                                             r[j], dp_data, plotSpots,iterations=iterations)
        
    return order_centers


def getSpotCenter(ky, kx, r, diff_data, plotSpot=False, iterations=1):
    """
    Find the center of mass (of pixel intensities) of a diffraction spot, 
    allowing for non-integer radii.
    """
    
    # Pad and round up to ensure the entire radius is accommodated
    pad_width = int(np.ceil(r))
    padded_data = np.pad(diff_data, pad_width=pad_width, mode='constant')

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

def circular_mask(center_y, center_x, radius):
    y, x = np.ogrid[-center_y:radius*2-center_y, -center_x:radius*2-center_x]
    mask = x**2 + y**2 <= radius**2
    return mask

def rem_negs(data):
    """
    Function to replace negative values in diffraction data
    """

    data[data < 1] = 1

    return data

def getIntensities(data, spots_order=None, center=None, r=6, coords=None):
    """
    Extract intensities from a single DP
    """

    # Determine how many intensities are to be extracted
    if type(spots_order) == int:
        if spots_order == 4:
            ints = np.zeros(18)
        elif spots_order < 4:
            ints = np.zeros(6)
    
        if center is None:
            center = getCenters(data, spots_order, r, coords)
    
        for int_idx, intensity in enumerate(ints):
    
            masked_data = data*make_mask(center[int_idx], r, mask_dim=(data.shape))
            ints[int_idx] = np.sum(masked_data[int(center[int_idx][0]-(r+1)):int(center[int_idx][0]+(r+1)),
                                               int(center[int_idx][1]-(r+1)):int(center[int_idx][1]+(r+1))])

    elif spots_order is None:
        
        num_ints = len(coords)
        ints = np.zeros(num_ints)
        
        for int_idx in range(num_ints):
            masked_data = data*make_mask(center[int_idx], r[int_idx], mask_dim=(data.shape))
            ints[int_idx] = np.sum(masked_data)
    
    return ints

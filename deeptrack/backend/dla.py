"Special functions for placing spheres in a volume"

import numpy as np
from ._config import cupy

def random_spheres(xy_offset_range, zrange, radius, output_region, z_upscale = 1e-6, seed = []):
    """ Simple function to add random spheres to a region of a volume. 
    The spheres are placed in a square region around the center according to the size of xy_offset_range.

    Just adds them randomly, and does not take into account wether they overlap etc.

    Parameters
    ----------
    xy_offset_range : tuple
        The range of x and y offsets to use for the spheres. I.e. how much to deviate from the center.
    zrange : tuple
        The range of z offsets to use for the spheres.
    radius : float or list
        The radius of the spheres.
    z_upscale : int or float
        The upsample factor in z, since z is defined in mikrometers.
    output_region : tuple
        The region of the volume to output the spheres.
    seed : int
        Random seed
    """
    #Set random seed if inputed
    if isinstance(seed, int):
        np.random.seed(seed)

    #Having offset as 0 will generate positions at the same place all the time.
    if xy_offset_range[0] == 0 and xy_offset_range[1] == 0:
        raise ValueError(f"xy_offset_range is {xy_offset_range}, all spheres will be placed in the center, change to greater value than 0.")

    n_spheres = len(radius)

    xSize =  output_region[2] - output_region[0]
    ySize =  output_region[3] - output_region[1]

    midpoint_x = xSize / 2
    midpoint_y = ySize / 2

    if midpoint_x-xy_offset_range[0]<0 or midpoint_x + xy_offset_range[0]>xSize or midpoint_y-xy_offset_range[1]<0 or midpoint_y + xy_offset_range[1]>ySize:
        raise ValueError(f"Variable xy_offset_range = {xy_offset_range} is to large! Change to smaller value! Should be less than image_size/2")

    row = np.random.uniform(midpoint_x - xy_offset_range[0], midpoint_x + xy_offset_range[0], n_spheres)
    col = np.random.uniform(midpoint_y - xy_offset_range[1], midpoint_y + xy_offset_range[1], n_spheres)
    z = np.random.uniform(zrange[0], zrange[1], n_spheres) / z_upscale # Since z_range is in mikrometer... so we need to convert it to voxel size.

    positions = [[ri, ci, zi] for ri, ci, zi in zip(row, col, z)]

    return positions

def side_by_side_spheres(pos, xy_offset_range, zrange, radius, voxel_size, output_region, z_upscale = 1e-6, resolution = 40, seed = []):
    """Function that places two spheres side by side. Sphere 1 will always be located in the middle of the volume (with some random offset)...
    The second sphere will be placed side by side randomly (radius0 + radius1) distance away from the first sphere.

    Parameters  
    ----------
    pos : tuple
        initialized positions. 
    xy_offset_range : tuple
        The range of x and y offsets to use for the spheres. I.e. how much to deviate from the center.
    zrange : tuple
        The range of z offsets to use for the spheres.
    radius : float or list
        The radius of the spheres.
    voxel_size : float
        The voxel size of the volume.
    z_upscale : int or float
        The upsample factor in z, since z is defined in mikrometers.
    output_region : tuple
        The region of the volume to output the spheres.
    seed : int
        Random seed    
    """
    #Set random seed
    if isinstance(seed, int):
        np.random.seed(seed)

    #Raise error if radius not of len 2
    if len(radius) != 2:
        raise ValueError("This function is meant for two spheres, change the radius list to have two values.")
    
    #Convert to numpy array if cupy.
    if isinstance(pos, cupy.ndarray):
        pos = cupy.asnumpy(pos)

    xSize =  output_region[2] - output_region[0]
    ySize =  output_region[3] - output_region[1]

    midpoint_x = xSize / 2
    midpoint_y = ySize / 2

    if midpoint_x-xy_offset_range[0]<0 or midpoint_x + xy_offset_range[0]>xSize or midpoint_y-xy_offset_range[1]<0 or midpoint_y + xy_offset_range[1]>ySize:
        raise ValueError(f"Variable xy_offset_range = {xy_offset_range} is to large! Change to smaller value! Should be less than image_size/2")   

    #If we have not predefined where the center of the aggregate shall be set it randomly within xy_offset_range.
    if pos[0]==xSize and pos[1]==ySize:
        row = np.random.uniform(midpoint_x - xy_offset_range[0], midpoint_x + xy_offset_range[0], 1)[0]
        col = np.random.uniform(midpoint_y - xy_offset_range[1], midpoint_y + xy_offset_range[1], 1)[0]
    else:
        row = pos[0]
        col = pos[1]

    z = np.random.uniform(zrange[0], zrange[1], 1)[0] / z_upscale # Since z_range is in mikrometer... so we need to convert it to voxel size.

    #First sphere position
    center = (row, col, z)
    
    #Second sphere position
    radius = radius[0] + radius[1]

    phi_array = np.linspace(0, 1, resolution) * 2 * np.pi
    theta_array = np.linspace(0, 1, resolution) * np.pi
    phi, theta = np.meshgrid(phi_array, theta_array) 

    x_coords = radius*np.sin(theta)*np.cos(phi) / voxel_size[0]
    y_coords = radius*np.sin(theta)*np.sin(phi) / voxel_size[1]
    z_coords = radius*np.cos(theta) / voxel_size[2]

    random_i = np.random.randint(len(x_coords))
    random_j = np.random.randint(len(y_coords))

    #Choose a random coordinate from the grid and add offset.
    x, y, z = (
        x_coords[random_i][random_j] + center[0], 
        y_coords[random_i][random_j] + center[1], 
        z_coords[random_i][random_j] + center[2]
        )

    #The positions of the spheres
    positions = [
            [center[0], center[1], center[2]],
            [x, y, z]
        ]
    #calculate distance between spheres
    #distance = np.sqrt(np.sum((np.array(positions[0]) - np.array(positions[1]))**2))
    #print(distance)
    #print(np.sum(radius)/voxel_size[0])
    #print(positions)
    #print(positions[0][2], positions[1][2])
    #print(voxel_size)

    return positions


def small_packing_off_spheres(pos, xy_offset_range, zrange, radius, voxel_size, output_region, z_upscale = 1e-6, resolution = 40, seed = [], max_count = 10000):
    """
    Function places n spheres side by side. One sphere will always be connected to one other. It works recursively, i.e. the previous sphere will be connected to the next one etc.
    
    OBS. Not so optimized right now, but it works decently fast when having condition for max_count

    Parameters  
    ----------
    pos : tuple, or list of tuples.
        initialized positions. 
    xy_offset_range : array
        The range of x and y offsets to use for the spheres. I.e. how much to deviate from the center.
    zrange : tuple
        The range of z offsets to use for the spheres.
    radius : float or list
        The radius of the spheres.
    voxel_size : float
        The voxel size of the volume.
    z_upscale : int or float
        The upsample factor in z, since z is defined in mikrometers.
    output_region : tuple
        The region of the volume to output the spheres.
    seed : int
        Random seed
    max_count : int
        Max criterion to not get stuck in while loop.
    """
    
    #Set random seed if preset
    if isinstance(seed, int):
        np.random.seed(seed)

    if len(radius) < 3 or len(radius)>10:
        raise ValueError("This function is meant for between 3-10 spheres, change the radius list to have more/less values")

    #Here we get two spheres that are side-by-side
    positions = side_by_side_spheres(pos, xy_offset_range, zrange, radius[:2], voxel_size, output_region, z_upscale)

    xSize =  output_region[2] - output_region[0]
    ySize =  output_region[3] - output_region[1]

    #Initialize matrices for adding rest of spheres.
    phi_array = np.linspace(0, 1, resolution) * 2 * np.pi
    theta_array = np.linspace(0, 1, resolution) * np.pi

    phi, theta = np.meshgrid(phi_array, theta_array) 

    #Lambda functions for precalculations
    x_coords = lambda radius: radius*np.sin(theta)*np.cos(phi) / voxel_size[0]
    y_coords = lambda radius: radius*np.sin(theta)*np.sin(phi) / voxel_size[1]
    z_coords = lambda radius: radius*np.cos(theta) / voxel_size[2]
    distance_formula = lambda r1, r2: np.sqrt(np.sum((np.array(r1) - np.array(r2))**2))

    tol = 1e-18
    global_break = False

    #Try to build an aggregate, add a sphere to the previously added sphere, if all conditions satisfied...
    for i, r in enumerate(radius[2:]):
        condition = True

        #Get the position of the sphere we are trying to add.
        radius_temp = r + radius[i+1] # Is connected to the previous sphere. i+1 because we start from the second sphere.
        x, y, z = x_coords(radius_temp), y_coords(radius_temp), z_coords(radius_temp)

        #Extract all the radiis left 
        radiuses_left = radius[0:i+2]

        count = 0
        while condition:
            count+=1

            #Get the coordinates for the sphere we are trying to add.
            p_i = np.random.randint(0, resolution, 1)
            p_j = np.random.randint(0, resolution, 1)

            #Current random particle position.
            point_particle = (
                x[p_i, p_j] + positions[-1][0], 
                y[p_i, p_j] + positions[-1][1], 
                z[p_i, p_j] + positions[-1][2]
                )

            comp_list = []
            for j, rr in enumerate(radiuses_left):
                
                #Check eucledian distance between our generated random particle position, and if its overlap with the any other spheres.
                if distance_formula(
                    point_particle,
                    (positions[j][0], positions[j][1], positions[j][2])
                    ) > ((rr + radius_temp) / voxel_size[0] - tol):
                    comp_list.append(1)
                else:
                    comp_list.append(0)

            #If condition true for all spheres, add the sphere.
            if np.sum(comp_list) == len(comp_list) and 0<point_particle[0]<xSize and 0<point_particle[1]<ySize:
                condition = False
                positions.append([float(point_particle[0]), float(point_particle[1]), float(point_particle[2])])
                break

            if count > max_count:
                #print("Max count reached... No more spheres can be added // relax max count condition and/or change radius list, or xy_offset_range")
                global_break = True
                break

        if global_break:
            break
        
    return positions
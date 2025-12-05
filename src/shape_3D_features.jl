using LinearAlgebra, Random

"""
    get_shape3d_features(mask::AbstractArray{<:Real, 3}, spacing::Vector{Float32}; verbose=false, keep_largest_only=true, pad_width=1, threshold=0)
    
    Extracts 3D shape features from the given mask.
    
    # Arguments
        - `mask`: A 3D mask array (any numeric type). Values > threshold are considered foreground.
        - `spacing`: A vector containing the voxel spacing in each dimension (x, y, z).
        - `verbose`: If true, prints progress messages.
        - `keep_largest_only`: If true, keeps only the largest connected component (default: true).
        - `pad_width`: Number of layers to pad around the mask for Marching Cubes (default: 1).
        - `threshold`: Threshold for binarization (default: 0.5). Values > threshold are set to true.
        
    # Returns
        - A dictionary containing the calculated 3D shape features.
        
    # Notes
        - The mask is automatically binarized using the specified threshold.
        - The mask is automatically padded to ensure correct surface extraction at boundaries.
        - By default, only the largest connected component is kept to ensure meaningful shape features.
        
    """
function get_shape3d_features(mask::AbstractArray{<:Real, 3}, spacing::Vector{Float32}; 
                               verbose=false, 
                               keep_largest_only=true,
                               pad_width=1,
                               threshold=0)

    if verbose
        println("Extracting 3D shape features...")
        println("Input mask size: $(size(mask))")
        println("Input mask type: $(eltype(mask))")
    end
    
    # Step 1: Binarize the mask
    if verbose
        println("Binarizing mask with threshold=$threshold...")
    end
    mask_bin = mask .> threshold
    n_foreground = sum(mask_bin)
    
    if n_foreground == 0
        error("Mask is empty after binarization (no voxels > $threshold)")
    end
    
    if verbose
        total_voxels = length(mask_bin)
        percentage = 100 * n_foreground / total_voxels
        println("Foreground voxels: $n_foreground / $total_voxels ($(round(percentage, digits=2))%)")
    end
    
    # Step 2: Keep only largest component if requested
    processed_mask = mask_bin
    if keep_largest_only
        if verbose
            println("Checking for multiple connected components...")
        end
        processed_mask = keep_largest_component(mask_bin)
        if verbose && sum(processed_mask) < sum(mask_bin)
            removed_voxels = sum(mask_bin) - sum(processed_mask)
            percentage = 100 * removed_voxels / sum(mask_bin)
            println("Removed $removed_voxels voxels ($(round(percentage, digits=2))%) from smaller components")
        end
    end
    
    # Step 3: Pad the mask for Marching Cubes
    if verbose
        println("Padding mask with $pad_width layers...")
    end
    processed_mask = pad_mask(processed_mask, pad_width)
    if verbose
        println("Padded mask size: $(size(processed_mask))")
    end

    shape_3d_features = Dict{String, Float32}()

    # Step 4: Extract surface mesh using Marching Cubes
    if verbose
        println("Running Marching Cubes algorithm...")
    end
    triangles = marching_cubes_surface(processed_mask, spacing)
    if verbose
        println("Generated $(length(triangles)) triangles")
    end
    
    # Step 5: Calculate surface-based features
    if verbose
        println("Calculating surface area and volume...")
    end
    area, meshvol = surface_and_volume(triangles)
    shape_3d_features["shape3d_surface_area"] = area
    shape_3d_features["shape3d_mesh_volume"] = meshvol
    shape_3d_features["shape3d_surface_volume_ratio"] = area / meshvol
    shape_3d_features["shape3d_sphericity"] = sphericity(meshvol, area)

    # Step 6: Calculate principal axes features
    if verbose
        println("  Calculating principal axes features...")
    end
    coords = get_voxel_coords(processed_mask, spacing)

    # Step 7: Maximum 3D Diameter
    maxdiam = maximum_3d_diameter(triangles, sample_rate=0.3) # Sample rate for accuracy
    shape_3d_features["shape3d_maximum_3d_diameter"] = maxdiam
    
    axes_lengths, elongation, flatness = principal_axes_features(coords)
    shape_3d_features["shape3d_major_axis_length"] = axes_lengths[3]
    shape_3d_features["shape3d_minor_axis_length"] = axes_lengths[2]
    shape_3d_features["shape3d_least_axis_length"] = axes_lengths[1]
    shape_3d_features["shape3d_elongation"] = elongation
    shape_3d_features["shape3d_flatness"] = flatness
    
    # Step 7: Calculate voxel-based volume
    shape_3d_features["shape3d_voxel_volume"] = voxel_volume(processed_mask, spacing)
    
    return shape_3d_features
end

const edge_to_vertex = [
    (1,2), (2,3), (3,4), (4,1),
    (5,6), (6,7), (7,8), (8,5),
    (1,5), (2,6), (3,7), (4,8)
]

const edgeTable = UInt16[
   0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]

const triTable = [(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1),
(3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1),
(3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1),
(3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1),
(9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1),
(9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
(2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1),
(8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1),
(9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
(4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1),
(3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1),
(1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1),
(4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1),
(4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1),
(9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
(5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1),
(2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1),
(9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
(0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
(2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1),
(10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1),
(4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1),
(5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1),
(5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1),
(9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1),
(0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1),
(1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1),
(10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1),
(8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1),
(2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1),
(7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1),
(9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1),
(2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1),
(11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1),
(9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1),
(5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1),
(11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1),
(11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
(1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1),
(9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1),
(5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1),
(2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
(0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
(5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1),
(6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1),
(3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1),
(6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1),
(5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1),
(1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
(10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1),
(6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1),
(8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1),
(7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1),
(3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
(5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1),
(0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1),
(9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1),
(8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1),
(5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1),
(0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1),
(6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1),
(10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1),
(10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1),
(8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1),
(1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1),
(3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1),
(0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1),
(10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1),
(3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1),
(6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1),
(9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1),
(8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1),
(3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1),
(6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1),
(0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1),
(10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1),
(10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1),
(2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1),
(7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1),
(7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1),
(2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1),
(1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1),
(11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1),
(8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1),
(0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1),
(7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
(10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
(2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
(6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1),
(7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1),
(2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1),
(1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1),
(10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1),
(10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1),
(0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1),
(7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1),
(6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1),
(8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1),
(9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1),
(6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1),
(4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1),
(10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1),
(8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1),
(0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1),
(1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1),
(8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1),
(10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1),
(4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1),
(10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
(5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
(11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1),
(9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
(6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1),
(7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1),
(3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1),
(7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1),
(9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1),
(3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1),
(6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1),
(9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1),
(1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1),
(4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1),
(7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1),
(6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1),
(3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1),
(0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1),
(6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1),
(0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1),
(11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1),
(6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1),
(5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1),
(9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1),
(1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1),
(1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1),
(10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1),
(0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1),
(5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1),
(10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1),
(11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1),
(9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1),
(7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1),
(2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1),
(8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1),
(9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1),
(9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1),
(1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1),
(9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1),
(9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1),
(5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1),
(0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1),
(10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1),
(2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1),
(0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1),
(0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1),
(9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1),
(5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1),
(3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1),
(5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1),
(8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1),
(0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1),
(9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1),
(0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1),
(1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1),
(3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1),
(4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1),
(9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1),
(11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1),
(11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1),
(2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1),
(9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1),
(3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1),
(1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1),
(4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1),
(4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1),
(0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1),
(3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1),
(3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1),
(0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1),
(9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1),
(1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
]

""" 
    Linearly interpolate the position where an isosurface cuts
    an edge between two vertices, each with their own scalar value.
    """
function vertex_interp(p1, p2, valp1, valp2, isolevel)
    if abs(valp2 - valp1) < 1e-8
        return p1
    end
    mu = (isolevel - valp1) / (valp2 - valp1)
    return p1 .+ mu .* (p2 .- p1)
end

"""
    Extract the surface mesh from a 3D binary mask using the Marching Cubes algorithm
    Returns a list of triangles, each represented as a tuple of three 3D points.
    """
function marching_cubes_surface(mask::BitArray{3}, spacing::Vector{Float32})
    nx, ny, nz = size(mask)
    triangles = Vector{NTuple{3,Vector{Float64}}}()
    corner_offsets = [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    for z in 1:nz-1, y in 1:ny-1, x in 1:nx-1
        vals = [mask[x+dx, y+dy, z+dz] ? 1.0 : 0.0 for (dx,dy,dz) in corner_offsets]
        coords = [[(x-1+dx)*spacing[1], (y-1+dy)*spacing[2], (z-1+dz)*spacing[3]] for (dx,dy,dz) in corner_offsets]
        cubeindex = 0
        for i in 1:8
            if vals[i] > 0.5
                cubeindex |= (1<<(i-1))
            end
        end
        if edgeTable[cubeindex+1] == 0
            continue
        end
        vertlist = Vector{Union{Nothing,Vector{Float64}}}(undef, 12)
        for e in 1:12
            if (edgeTable[cubeindex+1] & (1<<(e-1))) != 0
                a, b = edge_to_vertex[e]
                vertlist[e] = vertex_interp(coords[a], coords[b], vals[a], vals[b], 0.5)
            end
        end
        tri = triTable[cubeindex+1]
        i = 1
        while i <= 16 && tri[i] != -1
            a = vertlist[tri[i]+1]
            b = vertlist[tri[i+1]+1]
            c = vertlist[tri[i+2]+1]
            if a !== nothing && b !== nothing && c !== nothing
                push!(triangles, (a, b, c))
            end
            i += 3
        end
    end
    return triangles
end

"""
    Calculate surface area and volume from a list of triangles.
    Each triangle is represented as a tuple of three 3D points.
    #Arguments
    - triangles: Vector of triangles, each triangle is a tuple of three 3D points.
    #Returns
    - area: Surface area of the shape.
    - volume: Volume of the shape."""
function surface_and_volume(triangles)
    area = 0.0
    volume = 0.0
    for (a, b, c) in triangles
        ab = b .- a
        ac = c .- a
        area += 0.5 * norm(cross(ab, ac))
        volume += dot(a, cross(b, c)) / 6
    end
    return Float32(area), Float32(abs(volume))
end

"""Calculate the sphericity given volume and surface area.
    Sphericity = (36π * V²)^(1/3) / A
    # Arguments
    - volume: Volume of the shape.
    - area: Surface area of the shape.
    # Returns
    - sphericity: The sphericity value.
    """
function sphericity(volume, area)
    return (36 * pi * volume^2)^(1/3) / area
end

"""
    Get the 3D coordinates of all voxels set to true in the binary mask.
    # Returns a vector of 3D points.
    #Arguments
    - mask: 3D BitArray representing the binary mask.
    - spacing: Vector of Float32 representing the voxel spacing in each dimension.
    #Returns
    - coords: Vector of 3D points (each point is a Vector{Float64"""
function get_voxel_coords(mask::BitArray{3}, spacing::Vector{Float32})
    coords = Vector{Vector{Float64}}()
    for I in CartesianIndices(mask)
        if mask[I]
            idx = Tuple(I)
            push!(coords, [(idx[1]-1+0.5)*spacing[1], (idx[2]-1+0.5)*spacing[2], (idx[3]-1+0.5)*spacing[3]])
        end
    end
    coords
end

"""
    Calculate the principal axes lengths, elongation, and flatness from a set of 3D coordinates.
    # Arguments
    - coords: Vector of 3D points (each point is a Vector{Float64}).
    # Returns
    - axes_lengths: Vector of Float32 containing the lengths of the principal axes.
    - elongation: Float32 representing the elongation (sqrt(lambda2/lambda3)).
    - flatness: Float32 representing the flatness (sqrt(lambda1/lambda3))."""
function principal_axes_features(coords)
    n = length(coords)
    if n == 0 return zeros(Float32,3), NaN32, NaN32 end
    M = reduce(hcat, coords)
    mu = mean(M, dims=2)
    centered = M .- mu
    covmat = centered * centered' / n
    eigvals = sort(eigen(Symmetric(covmat)).values)
    lambda1, lambda2, lambda3 = eigvals
    axes_lengths = Float32.([4*sqrt(lambda) for lambda in eigvals])
    elongation = lambda3 > 0 && lambda2 > 0 ? sqrt(lambda2/lambda3) : NaN32
    flatness = lambda3 > 0 && lambda1 > 0 ? sqrt(lambda1/lambda3) : NaN32
    return axes_lengths, Float32(elongation), Float32(flatness)
end

"""Calculate the volume of the shape represented by the binary mask.
    # Arguments
    - mask: 3D BitArray representing the binary mask.
    - spacing: Vector of Float32 representing the voxel spacing in each dimension.
    # Returns
    - volume: Float32 representing the volume of the shape.
    """
function voxel_volume(mask, spacing)
    return Float32(count(mask) * spacing[1] * spacing[2] * spacing[3])
end

"""
    Calculate the maximum 3D diameter from surface mesh vertices.
    Maximum 3D diameter is defined as the largest pairwise Euclidean distance 
    between tumor surface mesh vertices (also known as Feret Diameter).
    
    # Arguments
    - triangles: Vector of triangles from Marching Cubes, each triangle is a tuple of three 3D points.
    - sample_rate: Fraction of vertices to sample (0.0 to 1.0). Default 1.0 uses all vertices.
    - min_samples: Minimum number of vertices to use, regardless of sample_rate. Default 100.
    
    # Returns
    - max_diameter: Float32 representing the maximum 3D diameter.
    
    # Notes
    - sample_rate = 1.0: exact calculation (slow for large meshes)
    - sample_rate = 0.1-0.3: good approximation with significant speedup, default 0.05
    - sample_rate = 0.05: fast approximation, may underestimate slightly
    """
function maximum_3d_diameter(triangles::Vector{NTuple{3,Vector{Float64}}}; 
                            sample_rate::Float64=0.05,
                            min_samples::Int=100)
    if isempty(triangles)
        return Float32(0.0)
    end
    
    # Extract unique vertices from triangles
    vertices = Vector{Vector{Float64}}()
    for (a, b, c) in triangles
        push!(vertices, a)
        push!(vertices, b)
        push!(vertices, c)
    end
    
    # Remove duplicates (vertices shared by multiple triangles)
    unique_vertices = unique(vertices)
    
    n = length(unique_vertices)
    if n < 2
        return Float32(0.0)
    end
    
    # Determine number of vertices to use
    n_sample = n
    if sample_rate < 1.0
        n_sample = max(min_samples, Int(ceil(n * sample_rate)))
        n_sample = min(n_sample, n)  # Can't sample more than available
    end
    
    # Sample vertices if needed
    sampled_vertices = unique_vertices
    if n_sample < n
        Random.seed!(42)  # For reproducibility
        indices = randperm(n)[1:n_sample]
        sampled_vertices = unique_vertices[indices]
    end
    
    max_dist = 0.0
    n_used = length(sampled_vertices)
    
    # Calculate maximum pairwise distance between sampled vertices
    for i in 1:n_used-1
        for j in i+1:n_used
            dist = norm(sampled_vertices[i] .- sampled_vertices[j])
            if dist > max_dist
                max_dist = dist
            end
        end
    end
    
    return Float32(max_dist)
end


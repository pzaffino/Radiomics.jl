using LinearAlgebra

# Global constants — allocated once at module load, not at every function call
const LINE_TABLE_2D = NTuple{5,Int8}[
    (-1, -1, -1, -1, -1),
    ( 3,  0, -1, -1, -1),
    ( 0,  1, -1, -1, -1),
    ( 3,  1, -1, -1, -1),
    ( 1,  2, -1, -1, -1),
    ( 1,  2,  3,  0, -1),
    ( 0,  2, -1, -1, -1),
    ( 3,  2, -1, -1, -1),
    ( 2,  3, -1, -1, -1),
    ( 2,  0, -1, -1, -1),
    ( 0,  1,  2,  3, -1),
    ( 2,  1, -1, -1, -1),
    ( 1,  3, -1, -1, -1),
    ( 1,  0, -1, -1, -1),
    ( 0,  3, -1, -1, -1),
    (-1, -1, -1, -1, -1),
]

const VERT_LIST_2D = NTuple{2,Float64}[
    (0.0, 0.5), (0.5, 1.0), (1.0, 0.5), (0.5, 0.0)
]

const GRID_ANGLES_2D = ((0,0), (0,1), (1,1), (1,0))

const POINTS_EDGES_2D = ((0, 2), (3, 2))

"""
    Extract 2D shape features from a binary mask.   
    # Arguments
    - `mask_array`: 2D BitArray representing the binary mask of the shape.
    - `spacing`: Vector of Float64 representing the pixel spacing in each dimension.
    - `verbose`: Bool indicating whether to print progress messages.    
    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.
    """
function get_shape2d_features(mask_array::BitArray{2},
                               spacing::Vector{Float64};
                               verbose::Bool=false,
                               keep_largest_only::Bool=true)::Dict{String,Any}

    verbose && println("Extracting 2D shape features...")

    if keep_largest_only
        verbose && println("Checking for multiple connected components...")
        processed_mask, num_islands = keep_largest_component(mask_array)
        if verbose && sum(processed_mask) < sum(mask_array)
            removed = sum(mask_array) - sum(processed_mask)
            println("Removed $removed voxels ($(round(100*removed/sum(mask_array), digits=2))%) from smaller components")
        end
    else
        processed_mask = mask_array
        num_islands    = 1
    end

    shape_2d_features = Dict{String, Any}()

    perimeter, surface, diameter = get_coefficients(mask_array, spacing)

    # Perimeter
    shape_2d_features["shape2d_perimeter"] = perimeter

    # Mesh Surface
    shape_2d_features["shape2d_mesh_surface"] = surface

    # Maximum 2D Diameter
    shape_2d_features["shape2d_maximum_diameter"] = diameter

    # Pixel Surface
    shape_2d_features["shape2d_pixel_surface"] = get_pixel_surface(mask_array, spacing)

    # Perimeter Surface Ratio
    shape_2d_features["shape2d_perimeter_surface_ratio"] = get_perimeter_surface_ratio(perimeter, surface)

    # Sphericity
    shape_2d_features["shape2d_sphericity"] = get_sphericity(perimeter, surface)

    ev = get_eigenvalues(mask_array, spacing)

    # Major Axis Length
    shape_2d_features["shape2d_major_axis_length"] = get_major_axis_length(ev)

    # Minor Axis Length
    shape_2d_features["shape2d_minor_axis_length"] = get_minor_axis_length(ev)

    # Elongation
    shape_2d_features["shape2d_elongation"] = get_elongation(ev)

    # Number of Islands
    shape_2d_features["shape2d_number_of_islands"] = Int32(num_islands)

    return shape_2d_features
end

"""Calculate the perimeter to surface ratio.
    Perimeter Surface Ratio = Perimeter / Surface
    # Arguments
    - perimeter: Float64 representing the perimeter of the shape.
    - surface: Float64 representing the surface area of the shape.
    # Returns
    - Float64 representing the perimeter to surface ratio.
    """
function get_perimeter_surface_ratio(perimeter::Float64, surface::Float64)::Float64
    return perimeter / surface
end

"""Calculate the sphericity of a 2D shape.
    Sphericity = (2 * sqrt(π * Surface)) / Perimeter
    # Arguments
    - perimeter: Float64 representing the perimeter of the shape.
    - surface: Float64 representing the surface area of the shape.
    # Returns
    - Float64 representing the sphericity of the shape.
    """
function get_sphericity(perimeter::Float64, surface::Float64)::Float64
    return (2 * sqrt(pi * surface)) / perimeter
end

"""Calculate the eigenvalues of the covariance matrix of the shape's voxel coordinates.
    # Arguments
    - `mask`: 2D Boolean matrix representing the binary mask of the shape.
    - `spacing`: Vector of Float64 representing the pixel spacing in each dimension.
    # Returns
    - Vector of Float64 containing the eigenvalues sorted in ascending order.
    """
function get_eigenvalues(mask::AbstractMatrix{<:Bool}, spacing::Vector{Float64})::Vector{Float64}
    Np = count(mask)
    Np == 0 && return zeros(Float64, 2)

    xs = Vector{Float64}(undef, Np)
    ys = Vector{Float64}(undef, Np)
    k  = 1
    @inbounds for I in CartesianIndices(mask)
        if mask[I]
            xs[k] = (I[1] - 1) * spacing[1]
            ys[k] = (I[2] - 1) * spacing[2]
            k += 1
        end
    end

    meanx = sum(xs) / Np
    meany = sum(ys) / Np

    c11 = c12 = c22 = 0.0
    @inbounds for i in 1:Np
        dx = xs[i] - meanx
        dy = ys[i] - meany
        c11 += dx * dx
        c12 += dx * dy
        c22 += dy * dy
    end
    c11 /= Np; c12 /= Np; c22 /= Np

    ev = eigen(Symmetric([c11 c12; c12 c22])).values
    return Float64.(sort(ev, rev=false))
end

"""Calculate the pixel surface area of the shape represented by the binary mask.
    # Arguments
    - mask: 2D Boolean matrix representing the binary mask of the shape.
    - spacing: Vector of Float64 representing the pixel spacing in each dimension.
    # Returns
    - Float64 representing the pixel surface area of the shape.
    """
function get_pixel_surface(mask::AbstractMatrix{<:Bool}, spacing::Vector{Float64})::Float64
    return Float64(count(mask)) * spacing[1] * spacing[2]
end

"""
    Major axis length is calculated as 4 * sqrt(eigenvalue[2])
    where eigenvalue[2] is the second largest eigenvalue of the covariance matrix.
    This is a measure of the size of the shape in the direction of its major axis.
    """
function get_major_axis_length(ev::Vector{Float64})::Float64
    return ev[2] >= 0 ? 4.0 * sqrt(ev[2]) : NaN
end

"""
    Minor axis length is calculated as 4 * sqrt(eigenvalue[1])
    where eigenvalue[1] is the smallest eigenvalue of the covariance matrix.
    This is a measure of the size of the shape in the direction of its minor axis.
    """
function get_minor_axis_length(ev::Vector{Float64})::Float64
    return ev[1] >= 0 ? 4.0 * sqrt(ev[1]) : NaN
end

"""
    Elongation is calculated as the ratio of the major axis length to the minor axis length.
    This is a measure of how elongated the shape is.
    """
function get_elongation(ev::Vector{Float64})::Float64
    return sqrt(ev[1] / ev[2])
end

"""
Helper functions to calculate the maximum diameter of a 2D mesh and coefficients for the 2D shape
Original C code: https://github.com/AIM-Harvard/pyradiomics/blob/master/radiomics/src/cshape.c
"""
function calculate_mesh_diameter2d(points_flat::Vector{Float64})::Float64
    n = div(length(points_flat), 2)
    n < 2 && return 0.0

    # 1. Build the array of tuples
    pts = Vector{Tuple{Float64,Float64}}(undef, n)
    @inbounds for i in 1:n
        pts[i] = (points_flat[2i-1], points_flat[2i])
    end
    sort!(pts)

    # 2. Pre-allocate the hull array (maximum possible size: 2n)
    hull = Vector{Tuple{Float64,Float64}}(undef, 2n)
    k = 0

    # Lower hull
    @inbounds for i in 1:n
        p = pts[i]
        while k >= 2
            o = hull[k-1]
            a = hull[k]
            # Inline cross product to avoid function call overhead
            if (a[2]-o[2])*(p[1]-o[1]) - (a[1]-o[1])*(p[2]-o[2]) <= 0
                k -= 1
            else
                break
            end
        end
        k += 1
        hull[k] = p
    end

    # Upper hull
    t = k + 1
    @inbounds for i in n-1:-1:1
        p = pts[i]
        while k >= t
            o = hull[k-1]
            a = hull[k]
            if (a[2]-o[2])*(p[1]-o[1]) - (a[1]-o[1])*(p[2]-o[2]) <= 0
                k -= 1
            else
                break
            end
        end
        k += 1
        hull[k] = p
    end

    # 3. Compute maximum distance over hull vertices (h = k-1 to ignore duplicate point)
    h = k - 1
    max_dist2 = 0.0
    @inbounds for i in 1:h
        p1 = hull[i]
        for j in (i+1):h
            p2 = hull[j]
            dist2 = (p1[1]-p2[1])^2 + (p1[2]-p2[2])^2
            dist2 > max_dist2 && (max_dist2 = dist2)
        end
    end

    return sqrt(max_dist2)
end

"""Calculate perimeter, surface, and maximum diameter of a 2D shape represented by a binary mask.
    # Arguments
    - mask: 2D Integer matrix representing the binary mask of the shape.
    - spacing: Vector of Float64 representing the pixel spacing in each dimension.
    # Returns
    - A tuple containing perimeter, surface, and maximum diameter as Float64 values.
    """
function get_coefficients(mask::AbstractMatrix{<:Integer}, spacing::Vector{Float64})::Tuple{Float64,Float64,Float64}
    """
    The marching squares algorithm iterates from iy = 1 to ny - 1, so row 0 and row ny - 1
    are never used as the top-left corner of a cell.
    Without padding, all contour edges along the first and last row/column
    of the mask are never detected.

    Example with a (4,5) mask without padding:
    1 1 1 0 0
    1 1 1 1 1  ← row 0: NEVER visited as a top corner
    1 1 0 1 1
    1 1 1 1 1  ← row 3: NEVER visited as a top corner

    With zero padding:
    0 0 0 0 0 0 0
    0 1 1 1 0 0 0
    0 1 1 1 1 1 0  ← now row 1 (former row 0) has a zero above it
    0 1 1 0 1 1 0     and generates mixed cells detectable by the loop
    0 1 1 1 1 1 0
    0 0 0 0 0 0 0

    Each pixel of the original mask now has at least one zero neighbor on all sides,
    so all border cells are detected as mixed and the contour closes.

    Without this:
    - the shoelace formula does not receive a closed polygon => incorrect area
    - the diameter computation finds no valid vertices => returns 0.0
    """
    padded = zeros(Int, size(mask, 1) + 2, size(mask, 2) + 2)
    padded[2:end-1, 2:end-1] .= mask
    mask = padded

    perimeter = 0.0
    surface   = 0.0

    ny, nx = size(mask)
    vertices = Float64[]

    @inbounds for iy in 1:(ny-1)
        for ix in 1:(nx-1)
            square_idx = 0
            for a_idx in 1:4
                dy, dx = GRID_ANGLES_2D[a_idx]
                y = iy + dy
                x = ix + dx
                if 1 <= y <= ny && 1 <= x <= nx
                    if mask[y, x] != 0
                        square_idx |= 1 << (a_idx - 1)
                    end
                end
            end

            (square_idx == 0 || square_idx == 0xF) && continue

            t = 1
            while LINE_TABLE_2D[square_idx + 1][t * 2 - 1] >= 0
                va = LINE_TABLE_2D[square_idx + 1][t * 2 - 1] + 1
                vb = LINE_TABLE_2D[square_idx + 1][t * 2] + 1

                a1 = (iy - 1.0 + VERT_LIST_2D[va][1]) * spacing[1]
                a2 = (ix - 1.0 + VERT_LIST_2D[va][2]) * spacing[2]
                b1 = (iy - 1.0 + VERT_LIST_2D[vb][1]) * spacing[1]
                b2 = (ix - 1.0 + VERT_LIST_2D[vb][2]) * spacing[2]

                # Surface (cross product z)
                surface += (a1 * b2) - (b1 * a2)

                # Perimeter (Euclidean distance)
                perimeter += sqrt((a1 - b1)^2 + (a2 - b2)^2)

                t += 1
            end

            # Save points for diameter
            if square_idx > 7
                square_idx ⊻= 0xF
            end
            for t in 1:2
                if square_idx & (1 << POINTS_EDGES_2D[1][t]) != 0
                    push!(vertices, (iy - 1 + VERT_LIST_2D[POINTS_EDGES_2D[2][t] + 1][1]) * spacing[1])
                    push!(vertices, (ix - 1 + VERT_LIST_2D[POINTS_EDGES_2D[2][t] + 1][2]) * spacing[2])
                end
            end
        end
    end

    surface  = abs(surface) / 2.0
    diameter = calculate_mesh_diameter2d(vertices)
    return Float64(perimeter), Float64(surface), Float64(diameter)
end
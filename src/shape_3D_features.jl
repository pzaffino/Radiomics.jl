using LinearAlgebra
using Random
using Statistics

include("utils/shape_3D_features_lookup_tables.jl")

const Point3D    = NTuple{3, Float64}
const Triangle3D = NTuple{3, Point3D}

"""
    This function is used to interpolate the position of a vertex on an edge of a cube.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
@inline function vertex_interp_opt(p1::Point3D, p2::Point3D,
                                    valp1::Real, valp2::Real,
                                    isolevel::Float64)::Point3D
    if abs(isolevel - valp1) < 1e-5; return p1; end
    if abs(isolevel - valp2) < 1e-5; return p2; end
    if abs(valp2   - valp1) < 1e-5; return p1; end
    mu = (isolevel - valp1) / (valp2 - valp1)
    return (p1[1] + mu*(p2[1]-p1[1]),
            p1[2] + mu*(p2[2]-p1[2]),
            p1[3] + mu*(p2[3]-p1[3]))
end
"""
    This function is used to get the vertex on the edge of a cube.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
@inline function get_vert_on_edge(edge_idx::Int8,
                                   p0::Point3D, p1::Point3D, p2::Point3D, p3::Point3D,
                                   p4::Point3D, p5::Point3D, p6::Point3D, p7::Point3D,
                                   v0,v1,v2,v3,v4,v5,v6,v7,
                                   isolevel::Float64)::Point3D
    edge_idx == 0  && return vertex_interp_opt(p0, p1, v0, v1, isolevel)
    edge_idx == 1  && return vertex_interp_opt(p1, p2, v1, v2, isolevel)
    edge_idx == 2  && return vertex_interp_opt(p2, p3, v2, v3, isolevel)
    edge_idx == 3  && return vertex_interp_opt(p3, p0, v3, v0, isolevel)
    edge_idx == 4  && return vertex_interp_opt(p4, p5, v4, v5, isolevel)
    edge_idx == 5  && return vertex_interp_opt(p5, p6, v5, v6, isolevel)
    edge_idx == 6  && return vertex_interp_opt(p6, p7, v6, v7, isolevel)
    edge_idx == 7  && return vertex_interp_opt(p7, p4, v7, v4, isolevel)
    edge_idx == 8  && return vertex_interp_opt(p0, p4, v0, v4, isolevel)
    edge_idx == 9  && return vertex_interp_opt(p1, p5, v1, v5, isolevel)
    edge_idx == 10 && return vertex_interp_opt(p2, p6, v2, v6, isolevel)
    edge_idx == 11 && return vertex_interp_opt(p3, p7, v3, v7, isolevel)
    return p0
end
"""
    This function is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function marching_cubes_surface(mask::BitArray{3},
                                 spacing::Vector{Float64},
                                 isolevel::Float64=0.5)::Vector{Triangle3D}
    nx, ny, nz = size(mask)
    triangles    = Vector{Triangle3D}()
    sizehint!(triangles,   count(mask) * 2)

    @inbounds for z in 1:nz-1, y in 1:ny-1, x in 1:nx-1

        v0 = Float64(mask[x,   y,   z  ])
        v1 = Float64(mask[x+1, y,   z  ])
        v2 = Float64(mask[x+1, y+1, z  ])
        v3 = Float64(mask[x,   y+1, z  ])
        v4 = Float64(mask[x,   y,   z+1])
        v5 = Float64(mask[x+1, y,   z+1])
        v6 = Float64(mask[x+1, y+1, z+1])
        v7 = Float64(mask[x,   y+1, z+1])

        cubeindex = 0
        if v0 > isolevel; cubeindex |= 1;   end
        if v1 > isolevel; cubeindex |= 2;   end
        if v2 > isolevel; cubeindex |= 4;   end
        if v3 > isolevel; cubeindex |= 8;   end
        if v4 > isolevel; cubeindex |= 16;  end
        if v5 > isolevel; cubeindex |= 32;  end
        if v6 > isolevel; cubeindex |= 64;  end
        if v7 > isolevel; cubeindex |= 128; end

        (cubeindex == 0 || cubeindex == 255) && continue

        sx, sy, sz = spacing[1], spacing[2], spacing[3]
        x0, x1 = (x-1)*sx, x*sx
        y0, y1 = (y-1)*sy, y*sy
        z0, z1 = (z-1)*sz, z*sz

        p0 = (x0, y0, z0); p1 = (x1, y0, z0)
        p2 = (x1, y1, z0); p3 = (x0, y1, z0)
        p4 = (x0, y0, z1); p5 = (x1, y0, z1)
        p6 = (x1, y1, z1); p7 = (x0, y1, z1)

        edges_seq = casesClassic[cubeindex + 1]
        i = 1
        while i + 2 <= length(edges_seq)
            edges_seq[i] == -1 && break
            a = get_vert_on_edge(edges_seq[i],   p0,p1,p2,p3,p4,p5,p6,p7,
                                  v0,v1,v2,v3,v4,v5,v6,v7, isolevel)
            b = get_vert_on_edge(edges_seq[i+1], p0,p1,p2,p3,p4,p5,p6,p7,
                                  v0,v1,v2,v3,v4,v5,v6,v7, isolevel)
            c = get_vert_on_edge(edges_seq[i+2], p0,p1,p2,p3,p4,p5,p6,p7,
                                  v0,v1,v2,v3,v4,v5,v6,v7, isolevel)
            push!(triangles, (a, b, c))
            i += 3
        end
    end
    return triangles
end
"""
    This Function provide to extract the 2d diameter from the vertices
"""
function maximum_2d_diameters_from_vertices(verts::Vector{Point3D})::NTuple{3, Float64}
    n = length(verts)
    n < 2 && return (0.0, 0.0, 0.0)

    d_slice  = 0.0
    d_row    = 0.0
    d_column = 0.0

    @inbounds for i in 1:n-1
        a = verts[i]
        for j in i+1:n
            b = verts[j]
            dx = a[1]-b[1]; dy = a[2]-b[2]; dz = a[3]-b[3]
            dist2 = dx*dx + dy*dy + dz*dz
            a[3] == b[3] && dist2 > d_slice  && (d_slice  = dist2)
            a[2] == b[2] && dist2 > d_row    && (d_row    = dist2)
            a[1] == b[1] && dist2 > d_column && (d_column = dist2)
        end
    end

    return sqrt(d_slice), sqrt(d_row), sqrt(d_column)
end
"""
    This function is used to calculate the surface area and volume of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function calculate_mesh_metrics(triangles::Vector{Triangle3D})::Tuple{Float64, Float64}
    area   = 0.0
    volume = 0.0
    @inbounds for (p1, p2, p3) in triangles
        ax = p2[1]-p1[1]; ay = p2[2]-p1[2]; az = p2[3]-p1[3]
        bx = p3[1]-p1[1]; by = p3[2]-p1[2]; bz = p3[3]-p1[3]
        cx = ay*bz - az*by
        cy = az*bx - ax*bz
        cz = ax*by - ay*bx
        area += 0.5 * sqrt(cx*cx + cy*cy + cz*cz)
        volume += (p1[1]*(p2[2]*p3[3] - p2[3]*p3[2])
                 - p1[2]*(p2[1]*p3[3] - p2[3]*p3[1])
                 + p1[3]*(p2[1]*p3[2] - p2[2]*p3[1])) / 6.0
    end
    return area, abs(volume)
end
"""
    This function is used to calculate the sphericity of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function sphericity(area::Float64, volume::Float64)::Float64
    (area <= 0.0 || volume <= 0.0) && return 0.0
    return (π^(1/3) * (6.0 * volume)^(2/3)) / area
end
"""
    This function is used to calculate the maximum 3D diameter of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function maximum_3d_diameter(triangles::Vector{Triangle3D})::Float64

    isempty(triangles) && return 0.0

    verts = Vector{Point3D}(undef, length(triangles) * 3)
    k = 1
    @inbounds for (a, b, c) in triangles
        verts[k]   = a
        verts[k+1] = b
        verts[k+2] = c
        k += 3
    end
    unique!(verts)
    nv = length(verts)
    nv < 2 && return 0.0

    # Find the center of the bounding box
    min_x = min_y = min_z = Inf
    max_x = max_y = max_z = -Inf
    @inbounds for p in verts
        p[1] < min_x && (min_x = p[1])
        p[1] > max_x && (max_x = p[1])
        p[2] < min_y && (min_y = p[2])
        p[2] > max_y && (max_y = p[2])
        p[3] < min_z && (min_z = p[3])
        p[3] > max_z && (max_z = p[3])
    end
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2

    # Calculate the squared distance from the center for each point
    d2_center = Vector{Float64}(undef, nv)
    @inbounds for i in 1:nv
        p = verts[i]
        d2_center[i] = (p[1]-cx)^2 + (p[2]-cy)^2 + (p[3]-cz)^2
    end

    # Sort the points by distance from the center in descending order
    perm = sortperm(d2_center, rev=true)
    verts_sorted = verts[perm]
    d_center_sorted = sqrt.(d2_center[perm])

    # Search for the maximum diameter with Pruning (Triangle Inequality)
    max_sq = 0.0
    max_d = 0.0

    # Heuristics to get a good lower-bound (distances from the first point, the most external)
    p1 = verts_sorted[1]
    @inbounds for j in 2:nv
        p2 = verts_sorted[j]
        d2 = (p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2
        if d2 > max_sq
            max_sq = d2
            max_d = sqrt(max_sq)
        end
    end

    @inbounds for i in 1:nv-1
        # Outer pruning: if twice the distance of P_i from the center does not exceed the current diameter,
        # no subsequent point (being closer to the center) can ever form a larger diameter.
        if 2.0 * d_center_sorted[i] <= max_d
            break
        end

        pi = verts_sorted[i]
        for j in i+1:nv
            # Inner pruning: if the sum of the distances of P_i and P_j from the center does not exceed the diameter,
            # the same applies to all points P_k with k >= j.
            if d_center_sorted[i] + d_center_sorted[j] <= max_d
                break
            end

            pj = verts_sorted[j]
            d2 = (pi[1]-pj[1])^2 + (pi[2]-pj[2])^2 + (pi[3]-pj[3])^2
            if d2 > max_sq
                max_sq = d2
                max_d = sqrt(max_sq)
            end
        end
    end
    
    return max_d
end
"""
    This function is used to get the voxel coordinates of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function get_voxel_coords(mask::AbstractArray{Bool,3}, spacing::Vector{Float64})::Vector{Point3D}
    n      = count(mask)
    coords = Vector{Point3D}(undef, n)
    k      = 1
    @inbounds for I in CartesianIndices(mask)
        if mask[I]
            coords[k] = ((I[1]-1+0.5)*Float64(spacing[1]),
                         (I[2]-1+0.5)*Float64(spacing[2]),
                         (I[3]-1+0.5)*Float64(spacing[3]))
            k += 1
        end
    end
    return coords
end
"""
    This function is used to calculate the principal axes features of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function principal_axes_features(coords::Vector{Point3D})::Tuple{Vector{Float64}, Float64, Float64}
    n = length(coords)
    n < 2 && return zeros(Float64, 3), NaN, NaN

    mx = my = mz = 0.0
    @inbounds for p in coords; mx+=p[1]; my+=p[2]; mz+=p[3]; end
    mx/=n; my/=n; mz/=n

    sn = sqrt(Float64(n - 1))
    c11=c12=c13=c22=c23=c33 = 0.0
    @inbounds for p in coords
        dx=(p[1]-mx)/sn; dy=(p[2]-my)/sn; dz=(p[3]-mz)/sn
        c11+=dx*dx; c12+=dx*dy; c13+=dx*dz
        c22+=dy*dy; c23+=dy*dz; c33+=dz*dz
    end

    ev = sort(eigen(Symmetric([c11 c12 c13; c12 c22 c23; c13 c23 c33])).values)
    l1,l2,l3 = max(0.0,ev[1]), max(0.0,ev[2]), max(0.0,ev[3])

    axes  = [4*sqrt(l) for l in (l1,l2,l3)]
    elong = (l3>0 && l2>0) ? sqrt(l2/l3) : NaN
    flat  = (l3>0 && l1>0) ? sqrt(l1/l3) : NaN
    return axes, Float64(elong), Float64(flat)
end
"""
    This function is used to calculate the voxel volume of a mesh.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
function voxel_volume(mask::AbstractArray, spacing::Vector{Float64})::Float64
    return Float64(count(mask)) *
           Float64(spacing[1]) * Float64(spacing[2]) * Float64(spacing[3])
end
"""
    get_shape3d_features(mask::AbstractArray{<:Real, 3}, spacing::Vector{Float64}; verbose=false, keep_largest_only=true, pad_width=1, threshold=0)
    
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
function get_shape3d_features(mask::AbstractArray{<:Real,3},
                               spacing::Vector{Float64};
                               verbose::Bool           = false,
                               keep_largest_only::Bool = true,
                               pad_width::Int          = 1,
                               threshold::Float64      = 0.5)::Dict{String,Any}

    verbose && println("Extracting 3D shape features...")

    if keep_largest_only
        verbose && println("Checking for multiple connected components...")
        processed_mask, num_islands = keep_largest_component(mask)
        if verbose && sum(processed_mask) < sum(mask)
            removed = sum(mask) - sum(processed_mask)
            println("Removed $removed voxels ($(round(100*removed/sum(mask), digits=2))%) from smaller components")
        end
    else
        processed_mask = mask
        num_islands    = 1
    end

    processed_mask = pad_mask(processed_mask, pad_width)

    task_axes = Threads.@spawn begin
        verbose && println("[Thread 1] Calculating principal axes...")
        coords = get_voxel_coords(processed_mask, spacing)
        principal_axes_features(coords)
    end

    verbose && println("[Main Thread] Running Marching Cubes...")
    triangles = marching_cubes_surface(processed_mask, spacing)

    task_geom = Threads.@spawn begin
        verbose && println("[Thread 2] Calculating surface features...")
        loc_area, loc_vol = calculate_mesh_metrics(triangles)
        loc_ratio = loc_vol > 0 ? loc_area / loc_vol : 0.0
        loc_sph   = sphericity(loc_area, loc_vol)
        (loc_area, loc_vol, loc_ratio, loc_sph)
    end

    task_diam = Threads.@spawn begin
        verbose && println("[Thread 3] Calculating maximum diameter...")
        maximum_3d_diameter(triangles)
    end

    task_diam2d = Threads.@spawn begin
        verbose && println("[Thread 4] Calculating 2D diameters from mesh...")
        all_verts = Vector{Point3D}(undef, length(triangles) * 3)
        k = 1
        for (a, b, c) in triangles
            all_verts[k] = a; all_verts[k+1] = b; all_verts[k+2] = c
            k += 3
        end
        unique!(all_verts)
        maximum_2d_diameters_from_vertices(all_verts)
    end

    vol_voxel                          = voxel_volume(processed_mask, spacing)
    axes_lengths, elongation, flatness = fetch(task_axes)
    area, meshvol, vol_ratio, sph      = fetch(task_geom)
    maxdiam                            = fetch(task_diam)
    diam_slice, diam_column, diam_row = fetch(task_diam2d)

    shape_3d_features = Dict{String, Any}()

    shape_3d_features["shape3d_surface_area"]         = area
    shape_3d_features["shape3d_mesh_volume"]          = meshvol
    shape_3d_features["shape3d_surface_volume_ratio"] = vol_ratio
    shape_3d_features["shape3d_sphericity"]           = sph

    shape_3d_features["shape3d_maximum_3d_diameter"]  = maxdiam

    shape_3d_features["shape3d_least_axis_length"]    = axes_lengths[1]
    shape_3d_features["shape3d_minor_axis_length"]    = axes_lengths[2]
    shape_3d_features["shape3d_major_axis_length"]    = axes_lengths[3]
    shape_3d_features["shape3d_elongation"]           = elongation
    shape_3d_features["shape3d_flatness"]             = flatness

    shape_3d_features["shape3d_voxel_volume"]         = vol_voxel
    shape_3d_features["shape3d_number_of_islands"]    = Int32(num_islands)

    shape_3d_features["shape3d_maximum_2d_diameter_slice"]  = diam_slice
    shape_3d_features["shape3d_maximum_2d_diameter_row"]    = diam_row
    shape_3d_features["shape3d_maximum_2d_diameter_column"] = diam_column

    return shape_3d_features
end
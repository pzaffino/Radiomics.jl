"""
    marching_cubes_surface_gpu(mask::CuArray{Bool,3},
                               spacing::CuArray{Float64},
                               isolevel::Float64=0.5
                              )::Tuple{Vector{Triangle3D},CuArray{Triangle3D}}

    # Arguments
    - `mask::CuArray{Bool,3}`: Binary 3D mask stored on the GPU.
    - `spacing::CuArray{Float64}`: Physical voxel spacing.
    - `isolevel::Float64`: Threshold.

    # Returns
    - `Tuple{Vector{Triangle3D}, CuArray{Triangle3D}}`:
        - Triangle array stored on the CPU.
        - Triangle array stored on the GPU.
"""
function marching_cubes_surface_gpu(mask::CuArray{Bool,3},
    spacing::CuArray{Float64},
    isolevel::Float64=0.5)::Tuple{Vector{Triangle3D},CuArray{Triangle3D}}

    include(joinpath(
        @__DIR__,
        "..",
        "utils",
        "utils_gpu",
        "shape_3D_features_lookup_tables_gpu.jl"
    ))


    mask_length = length(mask)
    mask_size = size(mask)
    (Nx, Ny, Nz) = (mask_size[1] - 1, mask_size[2] - 1, mask_size[3] - 1)

    cube_indices = CUDA.zeros(Int, Nx, Ny, Nz)
    # how many triangles every voxel generates
    triangle_count = CUDA.zeros(Int, Nx, Ny, Nz)

    blocks_x = cld(Nx, CUDA_BLOCK_WIDTH_3D)
    blocks_y = cld(Ny, CUDA_BLOCK_HEIGHT_3D)
    blocks_z = cld(Nz, CUDA_BLOCK_DEPTH_3D)
    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) calculate_cubeindex!(mask, cube_indices, Nx, Ny, Nz, mask_length, isolevel)

    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) count_triangles!(cube_indices, triangle_count, casesClassic_gpu, Nx, Ny, Nz)

    counts = vec(triangle_count)
    triangles_idx = cumsum(counts) .- counts
    num_of_triangles = sum(triangle_count)
    triangles = CuArray{Triangle3D}(undef, num_of_triangles)

    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) generate_triangles!(mask, triangles, triangle_count, triangles_idx, cube_indices, spacing, casesClassic_gpu, Nx, Ny, Nz, num_of_triangles, isolevel)

    return Array(triangles), triangles

end

"""
    maximum_2d_diameters_from_vertices_gpu(
        verts::CuArray{Point3D}
    )::NTuple{3,Float64}

    Computes maximum vertex distances

    # Arguments
    - `verts::CuArray{Point3D}`: Mesh vertices stored on the GPU.

    # Returns
    - `NTuple{3,Float64}` containing:
"""
function maximum_2d_diameters_from_vertices_gpu(verts::CuArray{Point3D})::NTuple{3,Float64}
    n = length(verts)
    n < 2 && return (0.0, 0.0, 0.0)

    d_slice = CUDA.zeros(Float64, 1)
    d_row = CUDA.zeros(Float64, 1)
    d_column = CUDA.zeros(Float64, 1)

    blocks_x = cld(n, CUDA_BLOCK_WIDTH_2D)
    blocks_y = cld(n, CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (blocks_x, blocks_y) diam2d_kernel!(verts, d_slice, d_row, d_column, n)
    return sqrt(Array(d_slice)[1]), sqrt(Array(d_row)[1]), sqrt(Array(d_column)[1])
end

"""
    calculate_diam2d_gpu(triangles::CuArray{Triangle3D},
                         verbose::Bool)::Point3D

    Computes the maximum 2D diameters of a mesh on the GPU.

    # Arguments
    - `triangles::CuArray{Triangle3D}`: Mesh triangles stored on the GPU.
    - `verbose::Bool`:

    # Returns
    - `Point3D`: Maximum 2D diameters
"""
function calculate_diam2d_gpu(triangles::CuArray{Triangle3D}, verbose::Bool)::Point3D
    verbose && println("Calculating 2D diameters from mesh on the GPU...")

    num_triangles = length(triangles)
    all_verts = CuArray{Point3D}(undef, length(triangles) * 3)

    @cuda threads = CUDA_THREADS blocks = cld(num_triangles, CUDA_THREADS) all_verts_kernel!(triangles, all_verts, num_triangles)

    all_verts = unique_gpu(all_verts)

    diam2d = maximum_2d_diameters_from_vertices_gpu(all_verts)

    return diam2d
end

"""
    calculate_cubeindex!(mask::CuDeviceArray{Bool},
                         cubeindex::CuDeviceArray{Int},
                         Nx::Int,
                         Ny::Int,
                         Nz::Int,
                         mask_length::Int,
                         isolevel::Float64)

    Computes Marching Cubes cube indices for a binary volume.

    Each thread evaluates one voxel

    # Arguments
    - `mask`: Binary volume mask.
    - `cubeindex`: Output cube configuration indices.
    - `Nx`, `Ny`, `Nz`: Volume dimensions.
    - `mask_length`: Total number of voxels.
    - `isolevel`: Threshold used for classification.

    # Returns
    Returns `nothing`. Cube indices are stored directly on the GPU.
"""
function calculate_cubeindex!(mask::CuDeviceArray{Bool},
    cubeindex::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    mask_length::Int,
    isolevel::Float64)
    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > Nx || y > Ny || z > Nz
        return nothing
    end

    v0 = Float64(mask[x, y, z])
    v1 = Float64(mask[x+1, y, z])
    v2 = Float64(mask[x+1, y+1, z])
    v3 = Float64(mask[x, y+1, z])
    v4 = Float64(mask[x, y, z+1])
    v5 = Float64(mask[x+1, y, z+1])
    v6 = Float64(mask[x+1, y+1, z+1])
    v7 = Float64(mask[x, y+1, z+1])

    cubeindex[x, y, z] = 0
    if v0 > isolevel
        cubeindex[x, y, z] |= 1
    end
    if v1 > isolevel
        cubeindex[x, y, z] |= 2
    end
    if v2 > isolevel
        cubeindex[x, y, z] |= 4
    end
    if v3 > isolevel
        cubeindex[x, y, z] |= 8
    end
    if v4 > isolevel
        cubeindex[x, y, z] |= 16
    end
    if v5 > isolevel
        cubeindex[x, y, z] |= 32
    end
    if v6 > isolevel
        cubeindex[x, y, z] |= 64
    end
    if v7 > isolevel
        cubeindex[x, y, z] |= 128
    end

    return nothing
end

"""
    count_triangles!(cube_indices::CuDeviceArray{Int},
                     triangle_count::CuDeviceArray{Int},
                     casesClassic::CuDeviceArray,
                     Nx::Int,
                     Ny::Int,
                     Nz::Int)

    Counts the number of triangles generated

    # Arguments
    - `cube_indices`: Cube indices.
    - `triangle_count`: Output triangle counts.
    - `casesClassic`: Marching Cubes lookup table.
    - `Nx`, `Ny`, `Nz`: Volume dimensions.

    # Returns
    Returns `nothing`. Triangle counts are written directly on the GPU.
"""
function count_triangles!(cube_indices::CuDeviceArray{Int},
    triangle_count::CuDeviceArray{Int},
    casesClassic::CuDeviceArray,
    Nx::Int,
    Ny::Int,
    Nz::Int)
    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > Nx || y > Ny || z > Nz
        return nothing
    end

    triangle_sum = 0
    for k in 1:16
        val = casesClassic[cube_indices[x, y, z]+1, k]
        val != -1 ? triangle_sum += 1 : break
    end

    triangle_count[x, y, z] = Int(triangle_sum/3)

    return nothing

end

"""
    generate_triangles!(mask::CuDeviceArray{Bool},
                        triangles::CuDeviceArray{Triangle3D},
                        triangles_count::CuDeviceArray{Int},
                        triangles_idx::CuDeviceArray{Int},
                        cube_indices::CuDeviceArray{Int},
                        spacing::CuDeviceArray{Float64},
                        casesClassic::CuDeviceArray,
                        nx::Int,
                        ny::Int,
                        nz::Int,
                        num_triangles::Int,
                        isolevel::Float64)

    Generates Marching Cubes triangles from cube configurations.

    Each CUDA thread processes one cube, computes the vertices,
    and writes the resulting triangles into the output array.

    # Arguments
    - `mask`: Binary input volume.
    - `triangles`: Output triangle array.
    - `triangles_count`: Number of triangles per cube.
    - `triangles_idx`: Prefix sum offsets for triangle placement.
    - `cube_indices`: Marching Cubes cube configurations.
    - `spacing`: Physical voxel spacing.
    - `casesClassic`: Marching Cubes lookup table.
    - `nx`, `ny`, `nz`: Volume dimensions.
    - `num_triangles`: Total number of output triangles.
    - `isolevel`: Threshold.

    # Returns
    Returns `nothing`. Output triangle array is written directly on the GPU.
"""
function generate_triangles!(mask::CuDeviceArray{Bool},
    triangles::CuDeviceArray{Triangle3D},
    triangles_count::CuDeviceArray{Int},
    triangles_idx::CuDeviceArray{Int},
    cube_indices::CuDeviceArray{Int},
    spacing::CuDeviceArray{Float64},
    casesClassic::CuDeviceArray,
    nx::Int,
    ny::Int,
    nz::Int,
    num_triangles::Int,
    isolevel::Float64)

    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > nx || y > ny || z > nz
        return nothing
    end

    lin_idx = encode_xyz(x, y, z, nx, ny)

    v0 = Float64(mask[x, y, z])
    v1 = Float64(mask[x+1, y, z])
    v2 = Float64(mask[x+1, y+1, z])
    v3 = Float64(mask[x, y+1, z])
    v4 = Float64(mask[x, y, z+1])
    v5 = Float64(mask[x+1, y, z+1])
    v6 = Float64(mask[x+1, y+1, z+1])
    v7 = Float64(mask[x, y+1, z+1])

    sx, sy, sz = spacing[1], spacing[2], spacing[3]
    x0, x1 = (x - 1) * sx, x * sx
    y0, y1 = (y - 1) * sy, y * sy
    z0, z1 = (z - 1) * sz, z * sz

    p0 = (x0, y0, z0)
    p1 = (x1, y0, z0)
    p2 = (x1, y1, z0)
    p3 = (x0, y1, z0)
    p4 = (x0, y0, z1)
    p5 = (x1, y0, z1)
    p6 = (x1, y1, z1)
    p7 = (x0, y1, z1)

    cidx = cube_indices[x, y, z] + 1

    i = 1
    triangle_number = 0
    while i + 2 <= 16
        e1 = casesClassic[cidx, i]
        e1 == -1 && break
        e2 = casesClassic[cidx, i+1]
        e3 = casesClassic[cidx, i+2]

        a = get_vert_on_edge(e1, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)
        b = get_vert_on_edge(e2, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)
        c = get_vert_on_edge(e3, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)

        triangles[triangles_idx[lin_idx]+triangle_number+1] = (a, b, c)

        triangle_number += 1
        i += 3
    end

    return nothing
end

"""
    all_verts_kernel!(triangles::CuDeviceArray{Triangle3D},
                      all_verts::CuDeviceArray{Point3D},
                      num_triangles::Int)

    # Arguments
    - `triangles::CuDeviceArray`: Input triangle list.
    - `all_verts::CuDeviceArray`: Output vertex array.
    - `num_triangles::Int`: Number of triangles.

    # Returns
    Returns `nothing`. Vertices are stored directly on the GPU.
"""

function all_verts_kernel!(
    triangles::CuDeviceArray{Triangle3D},
    all_verts::CuDeviceArray{Point3D},
    num_triangles::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_triangles
        return nothing
    end

    a, b, c = triangles[i]
    k = 3i - 2

    all_verts[k] = a
    all_verts[k+1] = b
    all_verts[k+2] = c

    return nothing
end

"""
    diam2d_kernel!(verts::CuDeviceArray{Point3D},
                   d_slice::CuDeviceArray{Float64,1},
                   d_row::CuDeviceArray{Float64,1},
                   d_column::CuDeviceArray{Float64,1},
                   num_verts::Int)

    Computes maximum distances between vertices

    # Arguments
    - `verts::CuDeviceArray`: 
    - `d_slice::CuDeviceArray`: 
    - `d_row::CuDeviceArray`: 
    - `d_column::CuDeviceArray`:
    - `num_verts::Int`: Number of vertices.

    # Returns
    Returns `nothing`. Distance values are updated atomically on the GPU.
"""
function diam2d_kernel!(verts::CuDeviceArray{Point3D},
    d_slice::CuDeviceArray{Float64,1},
    d_row::CuDeviceArray{Float64,1},
    d_column::CuDeviceArray{Float64,1},
    num_verts)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_verts || j > num_verts || j <= i
        return nothing
    end

    a = verts[i]
    if j >= (i + 1) && j <= num_verts
        b = verts[j]
        dx = a[1] - b[1]
        dy = a[2] - b[2]
        dz = a[3] - b[3]
        dist2 = dx * dx + dy * dy + dz * dz

        if a[3] == b[3]
            CUDA.@atomic d_slice[1] = max(d_slice[1], dist2)
        end

        if a[2] == b[2]
            CUDA.@atomic d_row[1] = max(d_row[1], dist2)
        end

        if a[1] == b[1]
            CUDA.@atomic d_column[1] = max(d_column[1], dist2)
        end
    end

    return nothing
end
const Point3D = NTuple{3,Float64}
const Triangle3D = NTuple{3,Point3D}


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

    all_verts = CuArray(unique!(Array(all_verts)))

    diam2d = maximum_2d_diameters_from_vertices_gpu(all_verts)

    return diam2d
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
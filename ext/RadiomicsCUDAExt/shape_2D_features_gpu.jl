const LINE_TABLE_2D_GPU = (
    (-1, -1, -1, -1, -1),
    (3, 0, -1, -1, -1),
    (0, 1, -1, -1, -1),
    (3, 1, -1, -1, -1),
    (1, 2, -1, -1, -1),
    (1, 2, 3, 0, -1),
    (0, 2, -1, -1, -1),
    (3, 2, -1, -1, -1),
    (2, 3, -1, -1, -1),
    (2, 0, -1, -1, -1),
    (0, 1, 2, 3, -1),
    (2, 1, -1, -1, -1),
    (1, 3, -1, -1, -1),
    (1, 0, -1, -1, -1),
    (0, 3, -1, -1, -1),
    (-1, -1, -1, -1, -1),
)

const VERT_LIST_2D_GPU = (
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 0.5),
    (0.5, 0.0),
)

const GRID_ANGLES_2D_GPU = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
)

const POINTS_EDGES_2D_GPU = (
    (0, 2),
    (3, 2),
)

"""
    get_coefficients_gpu(mask::CuArray{Bool,2},
                         mask_indices::CuArray{Int},
                         spacing::CuArray{Float64}
                        )::Tuple{Float64,Float64,Float64}

    # Arguments
    - `mask`: Binary 2D mask stored on the GPU.
    - `spacing`: Pixel spacing.

    # Returns
    - `Tuple{Float64,Float64,Float64}`:
        - Perimeter
        - Surface
        - Diameter
"""
function get_coefficients_gpu(mask::CuArray{Bool,2},
    spacing::CuArray{Float64})::Tuple{Float64,Float64,Float64}

    padded = CUDA.zeros(Bool, size(mask, 1) + 2, size(mask, 2) + 2)
    padded[2:(end-1), 2:(end-1)] .= mask
    mask = padded

    ny, nx = size(mask)

    square_idxs = CUDA.zeros(Int, ny - 1, nx - 1)
    surface_global = CUDA.zeros(Float64, ny - 1, nx - 1)
    perimeter_global = CUDA.zeros(Float64, ny - 1, nx - 1)
    num_vertices = CUDA.zeros(Int, 1)

    max_points = 2 * (ny - 1) * (nx - 1)
    vertices = CUDA.zeros(Float64, 2 * max_points)

    blocks = (cld(nx - 1, CUDA_BLOCK_WIDTH_2D), cld(ny - 1, CUDA_BLOCK_HEIGHT_2D))

    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = blocks get_square_idxs!(
        mask, square_idxs, GRID_ANGLES_2D_GPU, nx, ny)

    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = blocks perimeter_surface_vertices_count!(
        square_idxs, surface_global, perimeter_global, vertices, spacing,
        ny, nx, LINE_TABLE_2D_GPU, VERT_LIST_2D_GPU, POINTS_EDGES_2D_GPU, num_vertices)

    perimeter = CUDA.sum(perimeter_global)
    surface = abs(CUDA.sum(surface_global)) / 2.0

    n = Array(num_vertices)[1]
    vertices_cpu = Array(vertices)[1:(2*n)]
    diameter = Radiomics.calculate_mesh_diameter2d(vertices_cpu, vertices)

    return Float64(perimeter), Float64(surface), Float64(diameter)
end

"""
    get_square_idxs!(mask::CuDeviceArray{Bool,2},
                     square_idxs::CuDeviceArray{Int},
                     GRID_ANGLES_2D,
                     nx::Int,
                     ny::Int)

    # Arguments
    - `mask`: Binary 2D mask stored on the GPU.
    - `square_idxs`: Array containing the marching squares configuration index for each element.
    - `GRID_ANGLES_2D`: Grid offsets.
    - `nx::Int`: Number of columns in the mask.
    - `ny::Int`: Number of rows in the mask.

    # Returns
    - `nothing`
"""
function get_square_idxs!(
    mask::CuDeviceArray{Bool,2},
    square_idxs::CuDeviceArray{Int},
    GRID_ANGLES_2D,
    nx::Int,
    ny::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx - 1 && j <= ny - 1
        square_idx = 0
        for a_idx in 1:4
            dy, dx = GRID_ANGLES_2D[a_idx]
            y = j + dy
            x = i + dx
            if 1 <= y <= ny && 1 <= x <= nx
                if mask[y, x] != 0
                    square_idx |= 1 << (a_idx - 1)
                end
            end
        end
        square_idxs[j, i] = square_idx
    end

    return nothing

end

"""
    perimeter_surface_vertices_count!(
        square_idxs::CuDeviceArray{Int},
        surface::CuDeviceArray{Float64},
        perimeter::CuDeviceArray{Float64},
        vertices::CuDeviceArray{Float64},
        spacing::CuDeviceArray{Float64},
        ny::Int,
        nx::Int,
        LINE_TABLE_2D,
        VERT_LIST_2D,
        POINTS_EDGES_2D,
        num_vertices::CuDeviceArray{Int}
    )

    # Arguments
    - `square_idxs`: Marching-squares configuration index for each element.
    - `surface`: array containing the surface contribution of each element.
    - `perimeter`: Array containing the perimeter contribution of each element.
    - `vertices`: Array storing the coordinates of vertices.
    - `spacing`: Pixel spacing
    - `ny::Int`: Number of rows in the mask.
    - `nx::Int`: Number of columns in the mask.
    - `LINE_TABLE_2D`: Lookup table
    - `VERT_LIST_2D`: Lookup table
    - `POINTS_EDGES_2D`: Lookup table
    - `num_vertices`: Counter for the number of vertices.

    # Returns
    - `nothing`:
"""
function perimeter_surface_vertices_count!(square_idxs::CuDeviceArray{Int},
    surface::CuDeviceArray{Float64},
    perimeter::CuDeviceArray{Float64},
    vertices::CuDeviceArray{Float64},
    spacing::CuDeviceArray{Float64},
    ny::Int,
    nx::Int,
    LINE_TABLE_2D,
    VERT_LIST_2D,
    POINTS_EDGES_2D,
    num_vertices::CuDeviceArray{Int})

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > nx - 1 || j > ny - 1
        return nothing
    end

    idx = square_idxs[j, i]

    surface_local = 0.0
    perimeter_local = 0.0

    t = 1
    while LINE_TABLE_2D[idx+1][t*2-1] >= 0
        va = LINE_TABLE_2D[idx+1][t*2-1] + 1
        vb = LINE_TABLE_2D[idx+1][t*2] + 1

        a1 = (j - 1.0 + VERT_LIST_2D[va][1]) * spacing[1]
        a2 = (i - 1.0 + VERT_LIST_2D[va][2]) * spacing[2]
        b1 = (j - 1.0 + VERT_LIST_2D[vb][1]) * spacing[1]
        b2 = (i - 1.0 + VERT_LIST_2D[vb][2]) * spacing[2]

        surface_local += (a1 * b2) - (b1 * a2)
        perimeter_local += sqrt((a1 - b1)^2 + (a2 - b2)^2)

        t += 1
    end

    surface[j, i] = surface_local
    perimeter[j, i] = perimeter_local
    point_idx = idx
    if point_idx > 7
        point_idx ⊻= 0xF
    end

    for t in 1:2
        if point_idx & (1 << POINTS_EDGES_2D[1][t]) != 0
            slot = CUDA.atomic_add!(pointer(num_vertices, 1), 1)
            vertices[2*slot+1] = (j - 1.0 + VERT_LIST_2D[POINTS_EDGES_2D[2][t]+1][1]) * spacing[1]
            vertices[2*slot+2] = (i - 1.0 + VERT_LIST_2D[POINTS_EDGES_2D[2][t]+1][2]) * spacing[2]
        end
    end

    return nothing
end

"""
    max_dist!(hull::CuDeviceArray,
              max_dist2::CuDeviceArray{Float64},
              h::Int)

    # Arguments
    - `hull`: Array containing the hull.
    - `max_dist2`: Single element GPU array storing the maximum squared distance between hull points.
    - `h::Int`: Number of points in the hull.

    # Returns
    - `nothing`
"""
function max_dist!(
    hull::CuDeviceArray,
    max_dist2::CuDeviceArray{Float64},
    h::Int
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > h || j > h || j <= i
        return nothing
    end

    p1 = hull[i]
    p2 = hull[j]

    dist2 = (p1[1] - p2[1])^2 + (p1[2] - p2[2])^2

    CUDA.@atomic max_dist2[1] = max(max_dist2[1], dist2)

    return nothing
end

function get_eigenvalues_gpu(
    mask::CuArray{Bool,2},
    mask_indices::CuArray{Int},
    spacing::CuArray{Float64})::Vector{Float64}

    Np = length(mask_indices)

    Np == 0 && return zeros(Float64, 2)

    xs = CuArray{Float64}(undef, Np)
    ys = CuArray{Float64}(undef, Np)

    nrows = size(mask, 1)

    blocks = cld(Np, CUDA_THREADS)

    @cuda threads=CUDA_THREADS blocks=blocks mask_coordinates!(mask_indices, spacing, xs, ys, nrows, Np)

    meanx = CUDA.sum(xs) / Np
    meany = CUDA.sum(ys) / Np

    c11 = CuArray([0.0])
    c12 = CuArray([0.0])
    c22 = CuArray([0.0])

    @cuda threads = CUDA_THREADS blocks=blocks eigen_kernel!(xs, ys, c11, c12, c22, meanx, meany, Np)

    c11_cpu = Array(c11)[1] / Np
    c12_cpu = Array(c12)[1] / Np
    c22_cpu = Array(c22)[1] / Np

    ev = eigvals(Symmetric([c11_cpu c12_cpu; c12_cpu c22_cpu]))
    return Float64.(sort(ev, rev=false))
end

function eigen_kernel!(xs::CuDeviceArray{Float64},
    ys::CuDeviceArray{Float64},
    c11::CuDeviceArray{Float64},
    c12::CuDeviceArray{Float64},
    c22::CuDeviceArray{Float64},
    meanx::Float64,
    meany::Float64,
    Np::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > Np
        return nothing
    end

    dx = xs[i] - meanx
    dy = ys[i] - meany
    CUDA.@atomic c11[1] += dx * dx
    CUDA.@atomic c12[1] += dx * dy
    CUDA.@atomic c22[1] += dy * dy

    return nothing
end

function mask_coordinates!(
    mask_indices::CuDeviceArray{Int},
    spacing::CuDeviceArray{Float64},
    xs::CuDeviceArray{Float64},
    ys::CuDeviceArray{Float64},
    nrows::Int,
    n::Int,
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > n
        return
    end

    idx = mask_indices[i]

    idx0 = idx - 1

    row = idx0 % nrows
    col = idx0 ÷ nrows

    xs[i] = row * spacing[1]
    ys[i] = col * spacing[2]

    return nothing
end

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

    TABLE DESCRIPTIONS:
    - `cases`:
        Maps the 256 possible cases to their respective "equivalence class"
        (base geometry) and specific variant/rotation. Useful for optimizing
        topological configurations and avoiding ambiguities.
    - `casesClassic_gpu` (Triangulation table):
        For each case, it defines which edges of the cube must be connected
        by vertices to form the mesh. Each triplet of values represents a
        triangle. The `-1` values indicate the end of the geometry for that cube.

    Source: LookUpTable.h — Thomas Lewiner, Math Dept, PUC-Rio
    (MarchingCubes 33 Algorithm, v0.2, 12/08/2002)
    Available in the scikit-image repository:
    https://github.com/scikit-image/scikit-image/blob/main/tools/precompute/mc_meta/LookUpTable.h
"""
function marching_cubes_surface_gpu(mask::CuArray{Bool,3},
    spacing::CuArray{Float64},
    isolevel::Float64=0.5)::Tuple{Vector{Triangle3D},CuArray{Triangle3D}}
    casesClassic_gpu = CuArray(
        Int8[
            -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 1 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 8 3 9 8 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 1 2 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 2 10 0 2 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            2 8 3 2 10 8 10 9 8 -1 -1 -1 -1 -1 -1 -1;
            3 11 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 11 2 8 11 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 9 0 2 3 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 11 2 1 9 11 9 8 11 -1 -1 -1 -1 -1 -1 -1;
            3 10 1 11 10 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 10 1 0 8 10 8 11 10 -1 -1 -1 -1 -1 -1 -1;
            3 9 0 3 11 9 11 10 9 -1 -1 -1 -1 -1 -1 -1;
            9 8 10 10 8 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 3 0 7 3 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 1 9 8 4 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 1 9 4 7 1 7 3 1 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 8 4 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 4 7 3 0 4 1 2 10 -1 -1 -1 -1 -1 -1 -1;
            9 2 10 9 0 2 8 4 7 -1 -1 -1 -1 -1 -1 -1;
            2 10 9 2 9 7 2 7 3 7 9 4 -1 -1 -1 -1;
            8 4 7 3 11 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            11 4 7 11 2 4 2 0 4 -1 -1 -1 -1 -1 -1 -1;
            9 0 1 8 4 7 2 3 11 -1 -1 -1 -1 -1 -1 -1;
            4 7 11 9 4 11 9 11 2 9 2 1 -1 -1 -1 -1;
            3 10 1 3 11 10 7 8 4 -1 -1 -1 -1 -1 -1 -1;
            1 11 10 1 4 11 1 0 4 7 11 4 -1 -1 -1 -1;
            4 7 8 9 0 11 9 11 10 11 0 3 -1 -1 -1 -1;
            4 7 11 4 11 9 9 11 10 -1 -1 -1 -1 -1 -1 -1;
            9 5 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 5 4 0 8 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 5 4 1 5 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            8 5 4 8 3 5 3 1 5 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 9 5 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 0 8 1 2 10 4 9 5 -1 -1 -1 -1 -1 -1 -1;
            5 2 10 5 4 2 4 0 2 -1 -1 -1 -1 -1 -1 -1;
            2 10 5 3 2 5 3 5 4 3 4 8 -1 -1 -1 -1;
            9 5 4 2 3 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 11 2 0 8 11 4 9 5 -1 -1 -1 -1 -1 -1 -1;
            0 5 4 0 1 5 2 3 11 -1 -1 -1 -1 -1 -1 -1;
            2 1 5 2 5 8 2 8 11 4 8 5 -1 -1 -1 -1;
            10 3 11 10 1 3 9 5 4 -1 -1 -1 -1 -1 -1 -1;
            4 9 5 0 8 1 8 10 1 8 11 10 -1 -1 -1 -1;
            5 4 0 5 0 11 5 11 10 11 0 3 -1 -1 -1 -1;
            5 4 8 5 8 10 10 8 11 -1 -1 -1 -1 -1 -1 -1;
            9 7 8 5 7 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 3 0 9 5 3 5 7 3 -1 -1 -1 -1 -1 -1 -1;
            0 7 8 0 1 7 1 5 7 -1 -1 -1 -1 -1 -1 -1;
            1 5 3 3 5 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 7 8 9 5 7 10 1 2 -1 -1 -1 -1 -1 -1 -1;
            10 1 2 9 5 0 5 3 0 5 7 3 -1 -1 -1 -1;
            8 0 2 8 2 5 8 5 7 10 5 2 -1 -1 -1 -1;
            2 10 5 2 5 3 3 5 7 -1 -1 -1 -1 -1 -1 -1;
            7 9 5 7 8 9 3 11 2 -1 -1 -1 -1 -1 -1 -1;
            9 5 7 9 7 2 9 2 0 2 7 11 -1 -1 -1 -1;
            2 3 11 0 1 8 1 7 8 1 5 7 -1 -1 -1 -1;
            11 2 1 11 1 7 7 1 5 -1 -1 -1 -1 -1 -1 -1;
            9 5 8 8 5 7 10 1 3 10 3 11 -1 -1 -1 -1;
            5 7 0 5 0 9 7 11 0 1 0 10 11 10 0 -1;
            11 10 0 11 0 3 10 5 0 8 0 7 5 7 0 -1;
            11 10 5 7 11 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            10 6 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 5 10 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 0 1 5 10 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 8 3 1 9 8 5 10 6 -1 -1 -1 -1 -1 -1 -1;
            1 6 5 2 6 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 6 5 1 2 6 3 0 8 -1 -1 -1 -1 -1 -1 -1;
            9 6 5 9 0 6 0 2 6 -1 -1 -1 -1 -1 -1 -1;
            5 9 8 5 8 2 5 2 6 3 2 8 -1 -1 -1 -1;
            2 3 11 10 6 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            11 0 8 11 2 0 10 6 5 -1 -1 -1 -1 -1 -1 -1;
            0 1 9 2 3 11 5 10 6 -1 -1 -1 -1 -1 -1 -1;
            5 10 6 1 9 2 9 11 2 9 8 11 -1 -1 -1 -1;
            6 3 11 6 5 3 5 1 3 -1 -1 -1 -1 -1 -1 -1;
            0 8 11 0 11 5 0 5 1 5 11 6 -1 -1 -1 -1;
            3 11 6 0 3 6 0 6 5 0 5 9 -1 -1 -1 -1;
            6 5 9 6 9 11 11 9 8 -1 -1 -1 -1 -1 -1 -1;
            5 10 6 4 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 3 0 4 7 3 6 5 10 -1 -1 -1 -1 -1 -1 -1;
            1 9 0 5 10 6 8 4 7 -1 -1 -1 -1 -1 -1 -1;
            10 6 5 1 9 7 1 7 3 7 9 4 -1 -1 -1 -1;
            6 1 2 6 5 1 4 7 8 -1 -1 -1 -1 -1 -1 -1;
            1 2 5 5 2 6 3 0 4 3 4 7 -1 -1 -1 -1;
            8 4 7 9 0 5 0 6 5 0 2 6 -1 -1 -1 -1;
            7 3 9 7 9 4 3 2 9 5 9 6 2 6 9 -1;
            3 11 2 7 8 4 10 6 5 -1 -1 -1 -1 -1 -1 -1;
            5 10 6 4 7 2 4 2 0 2 7 11 -1 -1 -1 -1;
            0 1 9 4 7 8 2 3 11 5 10 6 -1 -1 -1 -1;
            9 2 1 9 11 2 9 4 11 7 11 4 5 10 6 -1;
            8 4 7 3 11 5 3 5 1 5 11 6 -1 -1 -1 -1;
            5 1 11 5 11 6 1 0 11 7 11 4 0 4 11 -1;
            0 5 9 0 6 5 0 3 6 11 6 3 8 4 7 -1;
            6 5 9 6 9 11 4 7 9 7 11 9 -1 -1 -1 -1;
            10 4 9 6 4 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 10 6 4 9 10 0 8 3 -1 -1 -1 -1 -1 -1 -1;
            10 0 1 10 6 0 6 4 0 -1 -1 -1 -1 -1 -1 -1;
            8 3 1 8 1 6 8 6 4 6 1 10 -1 -1 -1 -1;
            1 4 9 1 2 4 2 6 4 -1 -1 -1 -1 -1 -1 -1;
            3 0 8 1 2 9 2 4 9 2 6 4 -1 -1 -1 -1;
            0 2 4 4 2 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            8 3 2 8 2 4 4 2 6 -1 -1 -1 -1 -1 -1 -1;
            10 4 9 10 6 4 11 2 3 -1 -1 -1 -1 -1 -1 -1;
            0 8 2 2 8 11 4 9 10 4 10 6 -1 -1 -1 -1;
            3 11 2 0 1 6 0 6 4 6 1 10 -1 -1 -1 -1;
            6 4 1 6 1 10 4 8 1 2 1 11 8 11 1 -1;
            9 6 4 9 3 6 9 1 3 11 6 3 -1 -1 -1 -1;
            8 11 1 8 1 0 11 6 1 9 1 4 6 4 1 -1;
            3 11 6 3 6 0 0 6 4 -1 -1 -1 -1 -1 -1 -1;
            6 4 8 11 6 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            7 10 6 7 8 10 8 9 10 -1 -1 -1 -1 -1 -1 -1;
            0 7 3 0 10 7 0 9 10 6 7 10 -1 -1 -1 -1;
            10 6 7 1 10 7 1 7 8 1 8 0 -1 -1 -1 -1;
            10 6 7 10 7 1 1 7 3 -1 -1 -1 -1 -1 -1 -1;
            1 2 6 1 6 8 1 8 9 8 6 7 -1 -1 -1 -1;
            2 6 9 2 9 1 6 7 9 0 9 3 7 3 9 -1;
            7 8 0 7 0 6 6 0 2 -1 -1 -1 -1 -1 -1 -1;
            7 3 2 6 7 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            2 3 11 10 6 8 10 8 9 8 6 7 -1 -1 -1 -1;
            2 0 7 2 7 11 0 9 7 6 7 10 9 10 7 -1;
            1 8 0 1 7 8 1 10 7 6 7 10 2 3 11 -1;
            11 2 1 11 1 7 10 6 1 6 7 1 -1 -1 -1 -1;
            8 9 6 8 6 7 9 1 6 11 6 3 1 3 6 -1;
            0 9 1 11 6 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            7 8 0 7 0 6 3 11 0 11 6 0 -1 -1 -1 -1;
            7 11 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            7 6 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 0 8 11 7 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 1 9 11 7 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            8 1 9 8 3 1 11 7 6 -1 -1 -1 -1 -1 -1 -1;
            10 1 2 6 11 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 3 0 8 6 11 7 -1 -1 -1 -1 -1 -1 -1;
            2 9 0 2 10 9 6 11 7 -1 -1 -1 -1 -1 -1 -1;
            6 11 7 2 10 3 10 8 3 10 9 8 -1 -1 -1 -1;
            7 2 3 6 2 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            7 0 8 7 6 0 6 2 0 -1 -1 -1 -1 -1 -1 -1;
            2 7 6 2 3 7 0 1 9 -1 -1 -1 -1 -1 -1 -1;
            1 6 2 1 8 6 1 9 8 8 7 6 -1 -1 -1 -1;
            10 7 6 10 1 7 1 3 7 -1 -1 -1 -1 -1 -1 -1;
            10 7 6 1 7 10 1 8 7 1 0 8 -1 -1 -1 -1;
            0 3 7 0 7 10 0 10 9 6 10 7 -1 -1 -1 -1;
            7 6 10 7 10 8 8 10 9 -1 -1 -1 -1 -1 -1 -1;
            6 8 4 11 8 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 6 11 3 0 6 0 4 6 -1 -1 -1 -1 -1 -1 -1;
            8 6 11 8 4 6 9 0 1 -1 -1 -1 -1 -1 -1 -1;
            9 4 6 9 6 3 9 3 1 11 3 6 -1 -1 -1 -1;
            6 8 4 6 11 8 2 10 1 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 3 0 11 0 6 11 0 4 6 -1 -1 -1 -1;
            4 11 8 4 6 11 0 2 9 2 10 9 -1 -1 -1 -1;
            10 9 3 10 3 2 9 4 3 11 3 6 4 6 3 -1;
            8 2 3 8 4 2 4 6 2 -1 -1 -1 -1 -1 -1 -1;
            0 4 2 4 6 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 9 0 2 3 4 2 4 6 4 3 8 -1 -1 -1 -1;
            1 9 4 1 4 2 2 4 6 -1 -1 -1 -1 -1 -1 -1;
            8 1 3 8 6 1 8 4 6 6 10 1 -1 -1 -1 -1;
            10 1 0 10 0 6 6 0 4 -1 -1 -1 -1 -1 -1 -1;
            4 6 3 4 3 8 6 10 3 0 3 9 10 9 3 -1;
            10 9 4 6 10 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 9 5 7 6 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 4 9 5 11 7 6 -1 -1 -1 -1 -1 -1 -1;
            5 0 1 5 4 0 7 6 11 -1 -1 -1 -1 -1 -1 -1;
            11 7 6 8 3 4 3 5 4 3 1 5 -1 -1 -1 -1;
            9 5 4 10 1 2 7 6 11 -1 -1 -1 -1 -1 -1 -1;
            6 11 7 1 2 10 0 8 3 4 9 5 -1 -1 -1 -1;
            7 6 11 5 4 10 4 2 10 4 0 2 -1 -1 -1 -1;
            3 4 8 3 5 4 3 2 5 10 5 2 11 7 6 -1;
            7 2 3 7 6 2 5 4 9 -1 -1 -1 -1 -1 -1 -1;
            9 5 4 0 8 6 0 6 2 6 8 7 -1 -1 -1 -1;
            3 6 2 3 7 6 1 5 0 5 4 0 -1 -1 -1 -1;
            6 2 8 6 8 7 2 1 8 4 8 5 1 5 8 -1;
            9 5 4 10 1 6 1 7 6 1 3 7 -1 -1 -1 -1;
            1 6 10 1 7 6 1 0 7 8 7 0 9 5 4 -1;
            4 0 10 4 10 5 0 3 10 6 10 7 3 7 10 -1;
            7 6 10 7 10 8 5 4 10 4 8 10 -1 -1 -1 -1;
            6 9 5 6 11 9 11 8 9 -1 -1 -1 -1 -1 -1 -1;
            3 6 11 0 6 3 0 5 6 0 9 5 -1 -1 -1 -1;
            0 11 8 0 5 11 0 1 5 5 6 11 -1 -1 -1 -1;
            6 11 3 6 3 5 5 3 1 -1 -1 -1 -1 -1 -1 -1;
            1 2 10 9 5 11 9 11 8 11 5 6 -1 -1 -1 -1;
            0 11 3 0 6 11 0 9 6 5 6 9 1 2 10 -1;
            11 8 5 11 5 6 8 0 5 10 5 2 0 2 5 -1;
            6 11 3 6 3 5 2 10 3 10 5 3 -1 -1 -1 -1;
            5 8 9 5 2 8 5 6 2 3 8 2 -1 -1 -1 -1;
            9 5 6 9 6 0 0 6 2 -1 -1 -1 -1 -1 -1 -1;
            1 5 8 1 8 0 5 6 8 3 8 2 6 2 8 -1;
            1 5 6 2 1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 3 6 1 6 10 3 8 6 5 6 9 8 9 6 -1;
            10 1 0 10 0 6 9 5 0 5 6 0 -1 -1 -1 -1;
            0 3 8 5 6 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            10 5 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            11 5 10 7 5 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            11 5 10 11 7 5 8 3 0 -1 -1 -1 -1 -1 -1 -1;
            5 11 7 5 10 11 1 9 0 -1 -1 -1 -1 -1 -1 -1;
            10 7 5 10 11 7 9 8 1 8 3 1 -1 -1 -1 -1;
            11 1 2 11 7 1 7 5 1 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 1 2 7 1 7 5 7 2 11 -1 -1 -1 -1;
            9 7 5 9 2 7 9 0 2 2 11 7 -1 -1 -1 -1;
            7 5 2 7 2 11 5 9 2 3 2 8 9 8 2 -1;
            2 5 10 2 3 5 3 7 5 -1 -1 -1 -1 -1 -1 -1;
            8 2 0 8 5 2 8 7 5 10 2 5 -1 -1 -1 -1;
            9 0 1 5 10 3 5 3 7 3 10 2 -1 -1 -1 -1;
            9 8 2 9 2 1 8 7 2 10 2 5 7 5 2 -1;
            1 3 5 3 7 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 8 7 0 7 1 1 7 5 -1 -1 -1 -1 -1 -1 -1;
            9 0 3 9 3 5 5 3 7 -1 -1 -1 -1 -1 -1 -1;
            9 8 7 5 9 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            5 8 4 5 10 8 10 11 8 -1 -1 -1 -1 -1 -1 -1;
            5 0 4 5 11 0 5 10 11 11 3 0 -1 -1 -1 -1;
            0 1 9 8 4 10 8 10 11 10 4 5 -1 -1 -1 -1;
            10 11 4 10 4 5 11 3 4 9 4 1 3 1 4 -1;
            2 5 1 2 8 5 2 11 8 4 5 8 -1 -1 -1 -1;
            0 4 11 0 11 3 4 5 11 2 11 1 5 1 11 -1;
            0 2 5 0 5 9 2 11 5 4 5 8 11 8 5 -1;
            9 4 5 2 11 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            2 5 10 3 5 2 3 4 5 3 8 4 -1 -1 -1 -1;
            5 10 2 5 2 4 4 2 0 -1 -1 -1 -1 -1 -1 -1;
            3 10 2 3 5 10 3 8 5 4 5 8 0 1 9 -1;
            5 10 2 5 2 4 1 9 2 9 4 2 -1 -1 -1 -1;
            8 4 5 8 5 3 3 5 1 -1 -1 -1 -1 -1 -1 -1;
            0 4 5 1 0 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            8 4 5 8 5 3 9 0 5 0 3 5 -1 -1 -1 -1;
            9 4 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 11 7 4 9 11 9 10 11 -1 -1 -1 -1 -1 -1 -1;
            0 8 3 4 9 7 9 11 7 9 10 11 -1 -1 -1 -1;
            1 10 11 1 11 4 1 4 0 7 4 11 -1 -1 -1 -1;
            3 1 4 3 4 8 1 10 4 7 4 11 10 11 4 -1;
            4 11 7 9 11 4 9 2 11 9 1 2 -1 -1 -1 -1;
            9 7 4 9 11 7 9 1 11 2 11 1 0 8 3 -1;
            11 7 4 11 4 2 2 4 0 -1 -1 -1 -1 -1 -1 -1;
            11 7 4 11 4 2 8 3 4 3 2 4 -1 -1 -1 -1;
            2 9 10 2 7 9 2 3 7 7 4 9 -1 -1 -1 -1;
            9 10 7 9 7 4 10 2 7 8 7 0 2 0 7 -1;
            3 7 10 3 10 2 7 4 10 1 10 0 4 0 10 -1;
            1 10 2 8 7 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 9 1 4 1 7 7 1 3 -1 -1 -1 -1 -1 -1 -1;
            4 9 1 4 1 7 0 8 1 8 7 1 -1 -1 -1 -1;
            4 0 3 7 4 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            4 8 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            9 10 8 10 11 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 0 9 3 9 11 11 9 10 -1 -1 -1 -1 -1 -1 -1;
            0 1 10 0 10 8 8 10 11 -1 -1 -1 -1 -1 -1 -1;
            3 1 10 11 3 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 2 11 1 11 9 9 11 8 -1 -1 -1 -1 -1 -1 -1;
            3 0 9 3 9 11 1 2 9 2 11 9 -1 -1 -1 -1;
            0 2 11 8 0 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            3 2 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            2 3 8 2 8 10 10 8 9 -1 -1 -1 -1 -1 -1 -1;
            9 10 2 0 9 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            2 3 8 2 8 10 0 1 8 1 10 8 -1 -1 -1 -1;
            1 10 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            1 3 8 9 1 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 9 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            0 3 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
            -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        ]
    )

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
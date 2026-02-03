using LinearAlgebra
using Random
using Statistics

const Point3D = NTuple{3, Float64}
const Triangle3D = NTuple{3, Point3D}

const CUBE_OFFSETS = (
    (0,0,0), (1,0,0), (1,1,0), (0,1,0),
    (0,0,1), (1,0,1), (1,1,1), (0,1,1)
)

const edgeTable = UInt16[
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
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
    This function is used to interpolate the position of a vertex on an edge of a cube.
    It is used by the marching cubes algorithm to create a surface mesh from a 3D binary mask.
"""
@inline function vertex_interp_opt(p1::Point3D, p2::Point3D, valp1::Real, valp2::Real, isolevel::Float64)::Point3D
    if abs(isolevel - valp1) < 1e-5; return p1; end
    if abs(isolevel - valp2) < 1e-5; return p2; end
    if abs(valp2 - valp1) < 1e-5; return p1; end

    mu = (isolevel - valp1) / (valp2 - valp1)
    return (
        p1[1] + mu * (p2[1] - p1[1]),
        p1[2] + mu * (p2[2] - p1[2]),
        p1[3] + mu * (p2[3] - p1[3])
    )
end
"""
    Extract the surface mesh from a 3D binary mask using the Marching Cubes algorithm
    Returns a list of triangles, each represented as a tuple of three 3D points.
"""
function marching_cubes_surface(mask::AbstractArray{Bool, 3}, spacing::Vector{<:Real})
    nx, ny, nz = size(mask)
    triangles = Vector{Triangle3D}()
    sizehint!(triangles, floor(Int, count(mask) * 2))

    sp_x, sp_y, sp_z = Float64(spacing[1]), Float64(spacing[2]), Float64(spacing[3])
    vertlist = Vector{Point3D}(undef, 12)
    isolevel = 0.5

    @inbounds for z in 1:nz-1
        z_coord = (z - 1) * sp_z
        for y in 1:ny-1
            y_coord = (y - 1) * sp_y
            for x in 1:nx-1
                x_coord = (x - 1) * sp_x

                # Calcolo indice cubo
                cubeindex = 0
                if mask[x, y, z]     cubeindex |= 1   end
                if mask[x+1, y, z]   cubeindex |= 2   end
                if mask[x+1, y+1, z] cubeindex |= 4   end
                if mask[x, y+1, z]   cubeindex |= 8   end
                if mask[x, y, z+1]   cubeindex |= 16  end
                if mask[x+1, y, z+1] cubeindex |= 32  end
                if mask[x+1, y+1, z+1] cubeindex |= 64  end
                if mask[x, y+1, z+1]   cubeindex |= 128 end

                if edgeTable[cubeindex+1] == 0
                    continue
                end

                # Coordinate fisiche
                p0 = (x_coord, y_coord, z_coord)
                p1 = (x_coord + sp_x, y_coord, z_coord)
                p2 = (x_coord + sp_x, y_coord + sp_y, z_coord)
                p3 = (x_coord, y_coord + sp_y, z_coord)
                p4 = (x_coord, y_coord, z_coord + sp_z)
                p5 = (x_coord + sp_x, y_coord, z_coord + sp_z)
                p6 = (x_coord + sp_x, y_coord + sp_y, z_coord + sp_z)
                p7 = (x_coord, y_coord + sp_y, z_coord + sp_z)

                v0 = mask[x, y, z] ? 1.0 : 0.0
                v1 = mask[x+1, y, z] ? 1.0 : 0.0
                v2 = mask[x+1, y+1, z] ? 1.0 : 0.0
                v3 = mask[x, y+1, z] ? 1.0 : 0.0
                v4 = mask[x, y, z+1] ? 1.0 : 0.0
                v5 = mask[x+1, y, z+1] ? 1.0 : 0.0
                v6 = mask[x+1, y+1, z+1] ? 1.0 : 0.0
                v7 = mask[x, y+1, z+1] ? 1.0 : 0.0

                edges = edgeTable[cubeindex+1]

                if (edges & 1) != 0; vertlist[1] = vertex_interp_opt(p0, p1, v0, v1, isolevel); end
                if (edges & 2) != 0; vertlist[2] = vertex_interp_opt(p1, p2, v1, v2, isolevel); end
                if (edges & 4) != 0; vertlist[3] = vertex_interp_opt(p2, p3, v2, v3, isolevel); end
                if (edges & 8) != 0; vertlist[4] = vertex_interp_opt(p3, p0, v3, v0, isolevel); end
                if (edges & 16) != 0; vertlist[5] = vertex_interp_opt(p4, p5, v4, v5, isolevel); end
                if (edges & 32) != 0; vertlist[6] = vertex_interp_opt(p5, p6, v5, v6, isolevel); end
                if (edges & 64) != 0; vertlist[7] = vertex_interp_opt(p6, p7, v6, v7, isolevel); end
                if (edges & 128) != 0; vertlist[8] = vertex_interp_opt(p7, p4, v7, v4, isolevel); end
                if (edges & 256) != 0; vertlist[9] = vertex_interp_opt(p0, p4, v0, v4, isolevel); end
                if (edges & 512) != 0; vertlist[10] = vertex_interp_opt(p1, p5, v1, v5, isolevel); end
                if (edges & 1024) != 0; vertlist[11] = vertex_interp_opt(p2, p6, v2, v6, isolevel); end
                if (edges & 2048) != 0; vertlist[12] = vertex_interp_opt(p3, p7, v3, v7, isolevel); end

                t_idx = 1
                while t_idx <= 16
                    idx1 = triTable[cubeindex+1][t_idx]
                    if idx1 == -1 break end

                    idx2 = triTable[cubeindex+1][t_idx+1]
                    idx3 = triTable[cubeindex+1][t_idx+2]

                    push!(triangles, (vertlist[idx1+1], vertlist[idx2+1], vertlist[idx3+1]))
                    t_idx += 3
                end
            end
        end
    end
    return triangles
end
"""
    Calculate the surface area and volume of a mesh of triangles.
    Returns the surface area and volume.
"""
function surface_and_volume(triangles::Vector{Triangle3D})
    area = 0.0
    volume = 0.0

    @inbounds for (a, b, c) in triangles
        # Lati
        ab_x, ab_y, ab_z = b[1]-a[1], b[2]-a[2], b[3]-a[3]
        ac_x, ac_y, ac_z = c[1]-a[1], c[2]-a[2], c[3]-a[3]

        # Cross product per area
        cp_x = ab_y * ac_z - ab_z * ac_y
        cp_y = ab_z * ac_x - ab_x * ac_z
        cp_z = ab_x * ac_y - ab_y * ac_x

        area += 0.5 * sqrt(cp_x^2 + cp_y^2 + cp_z^2)

        # Cross product per volume
        bc_x = b[2]*c[3] - b[3]*c[2]
        bc_y = b[3]*c[1] - b[1]*c[3]
        bc_z = b[1]*c[2] - b[2]*c[1]

        volume += (a[1]*bc_x + a[2]*bc_y + a[3]*bc_z) / 6.0
    end
    return Float64(area), Float64(abs(volume))
end
"""
    Calculate the sphericity of the shape represented by the binary mask.
    # Arguments
        - volume: Float32 representing the volume of the shape.
        - area: Float32 representing the surface area of the shape.
    # Returns
        - sphericity: Float32 representing the sphericity of the shape. 
"""
function sphericity(volume, area)
    if area == 0 return 0.0 end
    return (36 * pi * volume^2)^(1 / 3) / area
end
"""
    Calculate the maximum 3D diameter of the shape represented by the binary mask.
    # Arguments
        - triangles: Vector of Triangle3D representing the mesh of the shape.
        - sample_rate: Float32 representing the sampling rate of the vertices.
        - min_samples: Int representing the minimum number of samples.
    # Returns
        - maximum_3d_diameter: Float32 representing the maximum 3D diameter of the shape.
"""
function maximum_3d_diameter(triangles::Vector{Triangle3D}; 
                            sample_rate::Float64=0.03, 
                            min_samples::Int=100)
    if isempty(triangles) 
        return Float64(0.0) 
    end

    unique_verts_set = Set{Point3D}()
    sizehint!(unique_verts_set, length(triangles) * 3)
    
    @inbounds for (a, b, c) in triangles
        push!(unique_verts_set, a)
        push!(unique_verts_set, b)
        push!(unique_verts_set, c)
    end
    
    unique_vertices = collect(unique_verts_set)
    n = length(unique_vertices)

    if n < 2 
        return Float64(0.0) 
    end

    n_sample = n
    if sample_rate < 1.0
        n_sample = max(min_samples, Int(ceil(n * sample_rate)))
        n_sample = min(n_sample, n)  
    end

    sampled_vertices = unique_vertices
    if n_sample < n
        Random.seed!(42)  # Riproducibilità
        indices = randperm(n)[1:n_sample]
        sampled_vertices = unique_vertices[indices]
    end

    max_dist_sq = 0.0
    n_used = length(sampled_vertices)

    @inbounds for i in 1:n_used-1
        p1 = sampled_vertices[i]
        for j in i+1:n_used
            p2 = sampled_vertices[j]
            dx = p1[1] - p2[1]
            dy = p1[2] - p2[2]
            dz = p1[3] - p2[3]
            d_sq = dx*dx + dy*dy + dz*dz
            
            if d_sq > max_dist_sq
                max_dist_sq = d_sq
            end
        end
    end

    return Float64(sqrt(max_dist_sq))
end
"""
    Calculate the voxel coordinates of the shape represented by the binary mask.
    # Arguments
        - mask: 3D BitArray representing the binary mask.
        - spacing: Vector of Float32 representing the voxel spacing in each dimension.
    # Returns
        - coords: Vector of Point3D representing the coordinates of the shape.
"""
function get_voxel_coords(mask::AbstractArray{Bool, 3}, spacing::Vector{<:Real})
    n = count(mask)
    coords = Vector{Point3D}(undef, n)
    idx_counter = 1
    
    @inbounds for I in CartesianIndices(mask)
        if mask[I]
            coords[idx_counter] = (
                (I[1] - 1 + 0.5) * Float64(spacing[1]),
                (I[2] - 1 + 0.5) * Float64(spacing[2]),
                (I[3] - 1 + 0.5) * Float64(spacing[3])
            )
            idx_counter += 1
        end
    end
    return coords
end
"""
    Calculate the principal axes features of the shape represented by the binary mask.
    # Arguments
        - coords: Vector of Point3D representing the coordinates of the shape.
    # Returns
        - axes_lengths: Vector of Float32 representing the lengths of the principal axes.
        - elongation: Float32 representing the elongation of the shape.
        - flatness: Float32 representing the flatness of the shape.
"""
function principal_axes_features(coords::Vector{Point3D})
    n = length(coords)
    if n < 2
        return zeros(Float64, 3), NaN, NaN
    end

    # 1. Calcola media
    mx, my, mz = 0.0, 0.0, 0.0
    @inbounds for p in coords
        mx += p[1]
        my += p[2]
        mz += p[3]
    end
    mx /= n
    my /= n
    mz /= n

    # 2. Calcola covarianza NORMALIZZATA per sqrt(N) (come PyRadiomics!)
    sqrt_n = sqrt(Float64(n))
    c11, c12, c13 = 0.0, 0.0, 0.0
    c22, c23, c33 = 0.0, 0.0, 0.0

    @inbounds for p in coords
        # CRITICO: Dividi per sqrt(n) durante la centratura
        dx = (p[1] - mx) / sqrt_n
        dy = (p[2] - my) / sqrt_n
        dz = (p[3] - mz) / sqrt_n
        
        c11 += dx*dx
        c12 += dx*dy
        c13 += dx*dz
        c22 += dy*dy
        c23 += dy*dz
        c33 += dz*dz
    end

    # Matrice covarianza (già normalizzata sopra)
    covmat = [c11 c12 c13; c12 c22 c23; c13 c23 c33]

    # 3. Autovalori
    eigvals = sort(eigen(Symmetric(covmat)).values)

    # Assicura non-negativi (errori numerici)
    l1 = max(0.0, eigvals[1])
    l2 = max(0.0, eigvals[2])
    l3 = max(0.0, eigvals[3])

    # 4. Calcola lunghezze assi e ratios
    axes_lengths = [4 * sqrt(l) for l in (l1, l2, l3)]
    
    elongation = (l3 > 0 && l2 > 0) ? sqrt(l2 / l3) : NaN
    flatness = (l3 > 0 && l1 > 0) ? sqrt(l1 / l3) : NaN

    return axes_lengths, Float64(elongation), Float64(flatness)
end
"""
    Calculate the volume of the shape represented by the binary mask.
    # Arguments
        - mask: 3D BitArray representing the binary mask.
        - spacing: Vector of Float32 representing the voxel spacing in each dimension.
    # Returns
        - volume: Float32 representing the volume of the shape.
"""
function voxel_volume(mask, spacing::Vector{<:Real})
    return Float64(count(mask) * Float64(spacing[1]) * Float64(spacing[2]) * Float64(spacing[3]))
end
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
function get_shape3d_features(mask::AbstractArray{<:Real,3}, 
                              spacing::Vector{<:Real};
                              verbose=false,
                              keep_largest_only=true,
                              pad_width=1,
                              threshold=0.5,
                              sample_rate=0.03)

    # Convert spacing to Float64 explicitly
    spacing_f64 = convert(Vector{Float64}, spacing)
    
    if verbose
        println("Extracting 3D shape features...")
        println("Input mask size: $(size(mask))")
        println("Sample rate for diameter: $sample_rate")
    end

    mask_bin = mask .> threshold
    if sum(mask_bin) == 0
        error("Mask is empty after binarization")
    end

    processed_mask = mask_bin
    num_islands = 1
    if keep_largest_only
        if verbose
            println("Checking for multiple connected components...")
        end
        processed_mask, num_islands = keep_largest_component(mask_bin)
        if verbose && sum(processed_mask) < sum(mask_bin)
            removed_voxels = sum(mask_bin) - sum(processed_mask)
            percentage = 100 * removed_voxels / sum(mask_bin)
            println("Removed $removed_voxels voxels ($(round(percentage, digits=2))%) from smaller components")
        end
    end

    processed_mask = pad_mask(processed_mask, pad_width)

    shape_3d_features = Dict{String,Float64}()

    if verbose println("Running Marching Cubes...") end
    triangles = marching_cubes_surface(processed_mask, spacing_f64)
    if verbose println("Generated $(length(triangles)) triangles") end

    if verbose println("Calculating surface features...") end
    area, meshvol = surface_and_volume(triangles)
    shape_3d_features["shape3d_surface_area"] = area
    shape_3d_features["shape3d_mesh_volume"] = meshvol
    shape_3d_features["shape3d_surface_volume_ratio"] = (meshvol > 0) ? area / meshvol : 0.0
    shape_3d_features["shape3d_sphericity"] = sphericity(meshvol, area)

    if verbose println("Calculating maximum diameter (sample_rate=$sample_rate)...") end
    maxdiam = maximum_3d_diameter(triangles, sample_rate=sample_rate)
    shape_3d_features["shape3d_maximum_3d_diameter"] = maxdiam

    if verbose println("Calculating principal axes...") end
    coords = get_voxel_coords(processed_mask, spacing_f64)
    axes_lengths, elongation, flatness = principal_axes_features(coords)

    shape_3d_features["shape3d_least_axis_length"] = axes_lengths[1]
    shape_3d_features["shape3d_minor_axis_length"] = axes_lengths[2]
    shape_3d_features["shape3d_major_axis_length"] = axes_lengths[3]
    shape_3d_features["shape3d_elongation"] = elongation
    shape_3d_features["shape3d_flatness"] = flatness

    shape_3d_features["shape3d_voxel_volume"] = voxel_volume(processed_mask, spacing_f64)
    shape_3d_features["shape3d_number_of_islands"] = Float64(num_islands)

    return shape_3d_features
end
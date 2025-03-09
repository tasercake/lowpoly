import torch


def generate_delaunay_polygons(*, points: torch.Tensor) -> list[torch.Tensor]:
    """
    Uses a Bowyer-Watson approach to compute the Delaunay triangulation of
    the 2D input points. Returns a list of (3, 2) float tensors, each
    representing one triangle's vertices.

    Args:
        points: (N, 2) float tensor of 2D coordinates in [0,1] or any range.

    Returns:
        A list of (3, 2) float tensors. Each tensor has the coordinates
        of a single triangle's vertices.
    """
    # 1) Possibly reject degenerate cases (fewer than 3 unique points):
    unique_positions = set((float(x), float(y)) for x, y in points)
    if len(unique_positions) < 3:
        # Not enough points to form any triangle
        return []

    # 2) Run Bowyer-Watson to get a list of triangles in index form
    triangles = _bowyer_watson(points)
    print(f"{triangles=}")

    # 3) Convert each triple of indices into a (3,2) tensor
    result = []
    for tri in triangles:
        # tri is something like (i1, i2, i3)
        coords = torch.stack([points[i] for i in tri], dim=0)  # shape (3, 2)
        result.append(coords)
    return result


# ------------------------------------------------------------------
#               Bowyer-Watson Delaunay Implementation
# ------------------------------------------------------------------


def _bowyer_watson(points: torch.Tensor) -> list[tuple[float, float, float]]:
    """
    Computes Delaunay triangulation in 2D for the given points using
    the Bowyer-Watson algorithm. Returns a list of triangles, where
    each triangle is a tuple of (i1, i2, i3) referencing 'points'.
    """
    # Convert each row to (x, y, idx)
    pt_list: list[tuple[float, float, float]] = []
    for i in range(points.shape[0]):
        x, y = points[i, 0].item(), points[i, 1].item()
        pt_list.append((x, y, i))

    # 1) Compute bounding box
    min_x = min(p[0] for p in pt_list)
    max_x = max(p[0] for p in pt_list)
    min_y = min(p[1] for p in pt_list)
    max_y = max(p[1] for p in pt_list)

    dx = max_x - min_x
    dy = max_y - min_y

    # If dx or dy is 0, make them nonzero so we can form a valid super-triangle.
    if dx < 1e-12:
        dx = 1.0
    if dy < 1e-12:
        dy = 1.0

    delta = max(dx, dy) * 10.0

    # 2) Create super-triangle points (with negative indices)
    p1 = (min_x - delta, min_y - delta, -1)
    p2 = (min_x - delta, max_y + 2 * delta, -2)
    p3 = (max_x + 2 * delta, max_y + delta, -3)
    pt_list.extend([p1, p2, p3])

    # Triangles stored as tuples of point indices
    triangles = [(p1[2], p2[2], p3[2])]  # just one large super-triangle

    # 3) Insert each real point into the triangulation
    for p in pt_list:
        # skip the super-triangle's "virtual" points
        if p[2] < 0:
            continue
        triangles = _add_point_to_triangles(p, pt_list, triangles)

    # 4) Remove any triangles that reference the super-triangle's vertices
    super_ids = {p1[2], p2[2], p3[2]}
    final_tris: list[tuple[float, float, float]] = []
    for tri in triangles:
        if any(v in super_ids for v in tri):
            continue
        final_tris.append(tri)

    return final_tris


def _add_point_to_triangles(point, all_points, triangles):
    """
    Inserts 'point' into the existing triangulation, returning the updated
    list of triangles.
    """
    xP, yP, idxP = point
    bad_triangles = []

    # 1) Find all triangles whose circumcircle contains the new point
    for tri in triangles:
        if _in_circumcircle(xP, yP, tri[0], tri[1], tri[2], all_points):
            bad_triangles.append(tri)

    # 2) Identify the boundary edges of the 'bad' region (edges shared by only one bad triangle)
    edges = []
    for tri in bad_triangles:
        iA, iB, iC = tri
        edges.extend([(iA, iB), (iB, iC), (iC, iA)])

    # Count occurrences of each edge
    from collections import Counter

    edge_count = Counter()
    for e in edges:
        e_sorted = tuple(sorted(e))
        edge_count[e_sorted] += 1
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]

    # 3) Remove the bad triangles
    new_triangles = [t for t in triangles if t not in bad_triangles]

    # 4) Create new triangles by connecting the new point to each boundary edge
    for e in boundary_edges:
        iA, iB = e
        new_triangles.append((iA, iB, idxP))

    return new_triangles


def _in_circumcircle(xP, yP, iA, iB, iC, all_points):
    """
    Checks if (xP, yP) lies within the circumcircle of triangle (iA, iB, iC).
    Some references use det < 0.0 here instead of det > 0.0,
    depending on the orientation of your points.
    """
    xA, yA, _ = _get_point(iA, all_points)
    xB, yB, _ = _get_point(iB, all_points)
    xC, yC, _ = _get_point(iC, all_points)

    mat = [
        [xA - xP, yA - yP, (xA - xP) ** 2 + (yA - yP) ** 2],
        [xB - xP, yB - yP, (xB - xP) ** 2 + (yB - yP) ** 2],
        [xC - xP, yC - yP, (xC - xP) ** 2 + (yC - yP) ** 2],
    ]
    det = _det_3x3(mat)

    # If the result is empty, try flipping the sign to 'return det < 0.0'
    return det > 0.0


def _get_point(idx, pt_list):
    # Return the (x, y, idx) tuple for the given index
    for p in pt_list:
        if p[2] == idx:
            return p
    raise ValueError(f"Point index {idx} not found")


def _det_3x3(m):
    """
    Computes determinant of a 3x3 matrix in plain Python.
    """
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


# TODO: Implement `generate_voronoi_polygons`
def generate_voronoi_polygons(*, points: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Voronoi not implemented yet.")


def rescale_polygons(
    polygons: list[torch.Tensor], size: tuple[float, float]
) -> list[torch.Tensor]:
    """
    Rescales [0, 1]-normalized polygons to the specified (width, height).
    Each polygon is assumed to be a tensor of shape (N, 2), representing
    the polygon's exterior coordinates.

    Args:
        polygons: A list of (N,2) PyTorch float tensors, each in [0,1].
        size: (width, height) specifying the target scaling factors.

    Returns:
        A list of (N,2) float tensors with scaled coordinates.
    """
    width, height = size
    scaled_polygons = []
    for poly in polygons:
        # Multiply each (x,y) in the polygon by (width, height)
        scale = torch.tensor([width, height], dtype=poly.dtype, device=poly.device)
        scaled_poly = poly * scale
        scaled_polygons.append(scaled_poly)
    return scaled_polygons

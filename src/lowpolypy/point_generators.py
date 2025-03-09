import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from shapely import MultiPoint, Point
from torch.types import Number


def random_points(*, num_points: int) -> MultiPoint:
    """
    Generates a set of random 2D points uniformly distributed within a unit square.
    
    Args:
        num_points (int): The number of random points to generate.
    
    Returns:
        MultiPoint: An object containing the generated 2D points.
    """
    coordinates = np.random.rand(num_points, 2)
    points = MultiPoint(coordinates)
    return points


def conv_points(
    *,
    image: np.ndarray,
    num_points: int = 1000,
    num_filler_points: int = 50,
    weight_filler_points: bool = True,
) -> MultiPoint:
    """
    Generate a MultiPoint from an image using gradient-based sampling.
    
    This function converts an RGB image to grayscale and computes a gradient map using a custom convolution
    kernel. It normalizes the gradient magnitude to weight the random selection of points so that regions with
    higher contrast are more likely to be sampled, with coordinates scaled to the unit square. Optionally,
    filler points are also sampled from areas with lower gradient intensity using either weighted or uniform
    random sampling, and then combined with the primary points.
    
    Args:
        image: An RGB image as a NumPy array.
        num_points: The number of primary points to sample from high-gradient regions.
        num_filler_points: The number of additional filler points to sample from lower-gradient areas.
        weight_filler_points: If True, use inverse gradient weights when sampling filler points; otherwise,
            sample them uniformly.
    
    Returns:
        A MultiPoint object containing the sampled points with normalized coordinates.
    """
    points: list[Point] = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array(
        [
            [-3 - 3j, 0 - 10j, +3 - 3j],
            [-10 + 0j, 0 + 0j, +10 + 0j],
            [-3 + 3j, 0 + 10j, +3 + 3j],
        ]
    )
    grad = signal.convolve2d(gray, kernel, boundary="symm", mode="same")
    mag = np.absolute(grad)
    mag = mag / mag.max()
    mag[mag <= 0.1] = 0
    mag = (mag * 255).astype(np.uint8)
    mag = cv2.equalizeHist(mag)
    weights = np.ravel(mag.astype(np.float32) / mag.sum())
    coordinates = np.arange(0, weights.size, dtype=np.uint32)
    choices = np.random.choice(coordinates, size=num_points, replace=False, p=weights)
    raw_points = np.unravel_index(choices, image.shape[:2])
    conv_points = np.stack(raw_points, axis=-1) / image.shape[:2]
    points.extend(MultiPoint(conv_points[..., ::-1]).geoms)

    if num_filler_points:
        inverse = 255 - cv2.dilate(mag, np.ones((5, 5), np.uint8), iterations=3)
        inverse = cv2.blur(inverse, ksize=(13, 13))
        weights = np.ravel(inverse.astype(np.float32) / inverse.sum())
        coordinates = np.arange(0, weights.size, dtype=np.uint32)
        choices = np.random.choice(
            coordinates,
            size=num_filler_points,
            replace=False,
            p=weights if weight_filler_points else None,
        )
        raw_points = np.unravel_index(choices, image.shape[:2])
        filler_points = np.stack(raw_points, axis=-1) / image.shape[:2]
        points.extend(MultiPoint(filler_points[..., ::-1]).geoms)

    return MultiPoint(points)


def select_points_sobel(image: torch.Tensor, N: int = 1000) -> torch.Tensor:
    """
    Select N points from the image based on Sobel gradient magnitude,
    returning [0, 1]-normalized (row, col) coordinates.

    Args:
        image: A PyTorch tensor of shape (H, W, 3) representing an RGB image.
        N: Number of points to sample.

    Returns:
        A float tensor of shape (N, 2) in [0,1], where each row is (row, col).
    """
    # 1. Convert to grayscale: shape => (H, W)
    gray = image.mean(dim=-1)

    # 2. Prepare for conv2d: shape => (1, 1, H, W)
    gray_4d = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 3. Define Sobel filters in X and Y
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    # 4. Apply convolution
    grad_x = F.conv2d(gray_4d, sobel_x, padding=1)  # shape => (1, 1, H, W)
    grad_y = F.conv2d(gray_4d, sobel_y, padding=1)  # shape => (1, 1, H, W)

    # 5. Gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze()  # shape => (H, W)

    # 6. Flatten and create a probability distribution
    H, W = grad_mag.shape
    grad_flat = grad_mag.view(-1)

    total = grad_flat.sum()
    if total <= 1e-12:
        # If the gradient is negligible, just sample uniformly
        probs = torch.ones_like(grad_flat) / grad_flat.numel()
    else:
        probs = grad_flat / total

    # 7. Sample N indices from the distribution
    indices = torch.multinomial(probs, N, replacement=False)

    # 8. Convert flat indices -> (row, col)
    rows = indices // W
    cols = indices % W
    coords = torch.stack([rows, cols], dim=1)  # shape => (N, 2)

    # 9. Normalize coordinates into [0,1] by dividing by (H-1) and (W-1).
    #    This ensures (0,0) -> (0.0,0.0) and (H-1,W-1) -> (1.0,1.0).
    coords = coords.to(dtype=torch.float32)
    if H > 1:
        coords[:, 0] /= H - 1
    if W > 1:
        coords[:, 1] /= W - 1

    return coords


def with_boundary_points(points: torch.Tensor) -> torch.Tensor:
    """
    Takes a set of 2D points in the unit square, computes their convex hull,
    then 'snaps' the hull vertices to the boundary [0,1]x[0,1], and
    finally appends the corners of the boundary. Returns a single (M, 2)
    float tensor of points (original + snapped + corners).

    Args:
        points: A (N, 2) float tensor of 2D coordinates, each in [0,1].

    Returns:
        A (M, 2) float tensor of updated points.
    """
    # 1) Compute the 2D convex hull
    hull_points = _convex_hull_2d(points)  # (H, 2)

    # 2) Snap each hull point onto the boundary of [0, 1]x[0, 1]
    snapped = []
    for i in range(hull_points.shape[0]):
        p = hull_points[i]
        snapped_pt = _closest_point_on_box_perimeter(p, 0.0, 1.0)
        snapped.append(snapped_pt)
    if len(snapped) > 0:
        snapped_tensor = torch.stack(snapped, dim=0)
    else:
        snapped_tensor = torch.empty((0, 2), dtype=torch.float32)

    # 3) Add the four corners of [0,1]x[0,1]
    corners = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # 4) Concatenate original points, snapped hull points, and corners
    return torch.cat([points, snapped_tensor, corners], dim=0)


def remove_duplicate_points(
    points: torch.Tensor, tolerance: float = 1e-3
) -> torch.Tensor:
    """
    Removes points that lie within 'tolerance' of another point in 2D.
    For each group of mutually close points, the first encountered is kept,
    the rest are discarded.

    Args:
        points: A (N, 2) float tensor of 2D coordinates.
        tolerance: Distance threshold to consider points 'duplicates'.

    Returns:
        A (K, 2) float tensor of filtered points (K <= N).
    """
    N = points.shape[0]
    if N <= 1:
        return points  # Nothing to remove

    # We'll gather all pairs (i,j) where dist < tolerance
    close_pairs = []
    tol_sq = tolerance * tolerance

    for i in range(N):
        # Once a point is discarded, we won't re-check it.
        for j in range(i + 1, N):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < tol_sq:
                close_pairs.append((i, j))

    # Build neighbor adjacency
    neighbors = {}
    for i, j in close_pairs:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)

    # Mark points as 'discarded' if they are neighbors of a point we keep
    discarded = set()
    for node in range(N):
        if node not in discarded:
            # keep 'node', discard all its neighbors
            discarded.update(neighbors.get(node, set()))

    # The final set of safe indices
    keep_indices = [i for i in range(N) if i not in discarded]

    return points[keep_indices]


def rescale_points(points: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Rescales [0, 1]-normalized points to the specified (width, height).
    The resulting coordinates are rounded to the nearest integer.

    Args:
        points: A (N, 2) float tensor of normalized (x, y) in [0,1].
        image_size: (width, height) specifying the target resolution.

    Returns:
        A (N, 2) float tensor of scaled coordinates (rounded).
    """
    width, height = image_size
    scale = torch.tensor([width, height], dtype=points.dtype, device=points.device)
    scaled_points = (points * scale).round()
    return scaled_points


# -----------------------------------------------------------------------
#      Helper functions: convex hull and boundary "snapping"
# -----------------------------------------------------------------------


def _convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D convex hull of a set of points using Andrew's monotone chain algorithm.
    
    This function expects a tensor of shape (N, 2) representing Cartesian coordinates.
    Duplicate points are removed; if fewer than three unique points remain, the input is
    returned directly. Otherwise, the convex hull is computed and returned as a (H, 2)
    float tensor with vertices in counter-clockwise order.
    
    Returns:
        torch.Tensor: A tensor of shape (H, 2) containing the convex hull vertices.
    """
    # Convert (N,2) into a Python list of (x, y) for easier sorting
    pts = [(points[i, 0].item(), points[i, 1].item()) for i in range(points.shape[0])]

    # Remove exact duplicates so the hull doesn't break
    pts = list(set(pts))
    if len(pts) < 3:
        # With < 3 points, the hull is just those points
        return torch.tensor(pts, dtype=torch.float32)

    # Sort by x, then by y
    pts.sort(key=lambda p: (p[0], p[1]))

    # Cross product of OA and OB vectors, >0 means left turn
    def cross(o, a, b):
        """
        Calculate the 2D cross product of vectors OA and OB.
        
        Given an origin point o and two points a and b, this function computes the cross
        product of the vectors (a - o) and (b - o), returning the scalar value corresponding
        to the signed area of the parallelogram defined by these vectors. A positive result
        indicates a counter-clockwise turn, a negative result indicates a clockwise turn,
        and zero indicates collinear points.
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower: list[tuple[Number, Number]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper: list[tuple[Number, Number]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last element of each list (it's the start point repeated)
    hull = lower[:-1] + upper[:-1]

    return torch.tensor(hull, dtype=torch.float32)


def _closest_point_on_box_perimeter(
    pt: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    """
    Finds the nearest point on the perimeter of an axis-aligned square.
    
    Given a 2D point and a square defined by the interval [low, high] for both axes, this
    function computes candidate points on the square's boundaries by clamping the point's
    coordinates and returns the candidate with the smallest Euclidean distance to the input.
    
    Parameters:
        pt (torch.Tensor): A tensor with two elements [x, y] representing a 2D point.
        low (float): The lower boundary of the square.
        high (float): The upper boundary of the square.
    
    Returns:
        torch.Tensor: The point on the square's perimeter that is closest to the input point.
    """
    # We want to “snap” (x, y) to whichever boundary edge is nearest.
    # The boundary is composed of four line segments, but we can just
    # compare the distance to the 4 possible "straight clamp" combos:
    #   (low, clamp(y)) , (high, clamp(y)), (clamp(x), low), (clamp(x), high)
    x, y = pt[0].item(), pt[1].item()
    cands = [
        torch.tensor([low, _clamp(y, low, high)]),
        torch.tensor([high, _clamp(y, low, high)]),
        torch.tensor([_clamp(x, low, high), low]),
        torch.tensor([_clamp(x, low, high), high]),
    ]

    best = None
    best_d = None
    for c in cands:
        dx = x - c[0].item()
        dy = y - c[1].item()
        dist_sq = dx * dx + dy * dy
        if (best_d is None) or (dist_sq < best_d):
            best_d = dist_sq
            best = c
    return best


def _clamp(v: float, low: float, high: float) -> float:
    """
    Clamps a value within the specified inclusive range.
    
    Returns the lower bound if the value is less than it, the upper bound if the value is
    greater than it, and the value itself otherwise.
    """
    return low if v < low else (high if v > high else v)

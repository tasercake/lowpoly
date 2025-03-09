import logging
import numpy as np
import torch

from .image_utils import resize_image
from .shaders import shade_kmeans

from .point_generators import (
    remove_duplicate_points,
    select_points_sobel,
    with_boundary_points,
)
from .polygon_generators import generate_delaunay_polygons, rescale_polygons

logger = logging.getLogger(__name__)


def run(
    *,
    image: np.ndarray,
    conv_points_num_points: int,
    conv_points_num_filler_points: int,
    weight_filler_points: bool,
    output_size: int,
) -> np.ndarray:
    # Run the pipeline
    """
    Processes an image to produce a polygon-based shaded output.
    
    This pipeline converts an input BGR image to an RGB tensor with normalized pixel values, then extracts
    key points using a Sobel filter. It augments the points with boundary values, removes near duplicates, and
    generates a Delaunay polygon mesh. The image is subsequently resized for performance, with polygons rescaled
    to the new dimensions, and a k-means based shading algorithm is applied to yield the final stylized image.
    
    Note:
        The conv_points_num_filler_points and weight_filler_points parameters are currently unused.
    
    Args:
        image: NumPy array representing the input image in BGR format.
        conv_points_num_points: The number of key points to extract from the image.
        conv_points_num_filler_points: Unused filler points count.
        weight_filler_points: Unused flag indicating whether to weight filler points.
        output_size: The target size for resizing the image prior to shading.
    
    Returns:
        NumPy array representing the shaded image.
    """
    logger.info("Computing points...")

    # Convert the BGR image numpy array to a RGB torch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    points = select_points_sobel(
        image=image_tensor,
        N=conv_points_num_points,
    )
    points = with_boundary_points(points)
    points = remove_duplicate_points(points, 3e-4)
    print(points, points.shape)

    logger.info("Generating polygon mesh from points...")
    polygons = generate_delaunay_polygons(points=points)
    print(polygons)

    logger.info("Shading polygons...")
    # Resize the image only before shading. This saves memory & compute when generating points and polygons.
    resized_image = resize_image(image=image, size=output_size)
    resized_polygons = rescale_polygons(
        polygons=polygons, size=(resized_image.shape[1] - 1, resized_image.shape[0] - 1)
    )
    shaded_image = shade_kmeans(image=resized_image, polygons=resized_polygons)

    return shaded_image

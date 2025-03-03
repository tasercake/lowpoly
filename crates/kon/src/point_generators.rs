use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use num_traits::NumCast;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

pub struct SobelResult<P> {
    pub raw_sobel_image: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub sobel_image: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub points: P,
}

/// Generate points based on the Sobel filter applied to an image.
///
/// # Arguments
/// * `image` - A reference to a DynamicImage that represents the input image.
/// * `num_points` - The number of points to sample randomly from the Sobel gradient image.
/// * `sharpness` - The sharpness of the Sobel gradient. Default is 1.0 for linear. >1.0 is more focused on edges. <1.0 is more random.
///
/// # Type Parameters
/// * `T` - The numeric type for point coordinates. Can be integers (signed/unsigned) or floats.
///
/// # Returns
/// * `SobelResult<Vec<(T, T)>>` - Contains the Sobel gradient image and a vector of sampled points of interest.
pub fn generate_points_from_sobel<T>(
    image: &DynamicImage,
    num_points: u32,
    sharpness: f32,
) -> SobelResult<Vec<(T, T)>>
where
    T: NumCast + Send,
{
    let (width, height) = image.dimensions();

    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();

    // Apply the Sobel filter to detect edges
    let sobel_image: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);

    // Raise each pixel's Sobel magnitude to `sharpness` power
    let pixel_weights = sobel_image.par_pixels().map(|p| {
        let mag = p[0] as f32;
        mag.powf(sharpness)
    });
    // Normalize the weights to the range [0, 1]
    let max_weight = pixel_weights.clone().max_by(|a, b| a.total_cmp(b)).unwrap();
    let pixel_weights: Vec<f32> = pixel_weights.map(|w| w / max_weight).collect();

    // Create a weighted distribution using the adjusted magnitudes.
    let dist = WeightedIndex::new(&pixel_weights)
        .expect("WeightedIndex failed: all weights were zero or invalid.");

    let points = (0..num_points).into_par_iter().map_init(
        || thread_rng(),
        move |rng, _| {
            let i = dist.sample(rng);
            let x = i as u32 % width;
            let y = i as u32 / width;
            (T::from(x).unwrap(), T::from(y).unwrap())
        },
    );

    SobelResult {
        raw_sobel_image: sobel_image,
        sobel_image: vec_to_imagebuffer(pixel_weights.clone(), width, height),
        points: points.collect(),
    }
}

/// Generate an iterator of `num_points` random points within the given dimensions.
/// # Arguments
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `num_points` - The number of random points to generate.
pub fn generate_random_points<T>(width: u32, height: u32, num_points: u32) -> Vec<(T, T)>
where
    T: NumCast + Send,
{
    (0..num_points)
        .into_par_iter()
        .map_init(
            || thread_rng(),
            move |rng, _| {
                (
                    T::from(rng.gen_range(0..width)).unwrap(),
                    T::from(rng.gen_range(0..height)).unwrap(),
                )
            },
        )
        .collect()
}

/// Convert a Vec<f32> into an ImageBuffer<Luma<u16>> by normalizing and scaling the values.
///
/// # Arguments
/// * `data` - A vector of f32 values representing the image.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Returns
/// * `ImageBuffer<Luma<u16>, Vec<u16>>` - The normalized image buffer.
fn vec_to_imagebuffer(data: Vec<f32>, width: u32, height: u32) -> ImageBuffer<Luma<u16>, Vec<u16>> {
    let min_val = data
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a, b| f64::min(a, b as f64));
    let max_val = data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a, b| f64::max(a, b as f64));

    let scaled_data: Vec<u16> = if max_val > min_val {
        data.into_iter()
            .map(|v| {
                (((v as f32 - min_val as f32) / (max_val as f32 - min_val as f32)) * 65535.0) as u16
            })
            .collect()
    } else {
        vec![0; (width * height) as usize]
    };

    ImageBuffer::from_raw(width, height, scaled_data)
        .expect("Failed to create ImageBuffer from Vec<u16>")
}

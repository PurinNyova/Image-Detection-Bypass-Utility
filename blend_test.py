import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import label
from PIL import Image
import argparse
import colorsys

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB array to HSV.
    
    Args:
        rgb (np.ndarray): Array of shape (N, 3) with RGB values in [0, 255].
    
    Returns:
        np.ndarray: Array of shape (N, 3) with HSV values (H in [0, 360], S and V in [0, 1]).
    """
    rgb = rgb / 255.0
    hsv = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        h, s, v = colorsys.rgb_to_hsv(rgb[i, 0], rgb[i, 1], rgb[i, 2])
        hsv[i] = [h * 360, s, v]
    return hsv

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert HSV array to RGB.
    
    Args:
        hsv (np.ndarray): Array of shape (N, 3) with HSV values (H in [0, 360], S and V in [0, 1]).
    
    Returns:
        np.ndarray: Array of shape (N, 3) with RGB values in [0, 255].
    """
    rgb = np.zeros_like(hsv)
    for i in range(hsv.shape[0]):
        r, g, b = colorsys.hsv_to_rgb(hsv[i, 0] / 360, hsv[i, 1], hsv[i, 2])
        rgb[i] = [r * 255, g * 255, b * 255]
    return rgb

def blend_colors(image: np.ndarray, tolerance: float = 10.0, min_region_size: int = 50) -> np.ndarray:
    """
    Blend similar and connected colors in an image into a single color with a slight random shift in HSV.
    
    Args:
        image (np.ndarray): Input image as a NumPy array (H, W, C) with uint8 values.
        tolerance (float): Tolerance for color similarity (Euclidean distance in RGB space).
                          Lower values mean stricter grouping.
        min_region_size (int): Minimum number of pixels in a connected region to process.
    
    Returns:
        np.ndarray: Output image with blended colors, same shape and dtype as input.
    """
    # Validate input
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array with uint8 dtype (H, W, C)")
    
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels).astype(np.float32)
    
    # Step 1: Create a mask for similar colors
    num_clusters = max(1, int(256 / tolerance))  # Rough estimate for initial clusters
    centroids, labels = kmeans2(pixels, num_clusters, minit='points')
    
    # Reshape labels to image shape
    label_map = labels.reshape(height, width)
    
    # Step 2: Apply connected component labeling for each cluster
    output_image = image.copy()  # Initialize output
    for cluster_id in range(num_clusters):
        # Create binary mask for current cluster
        cluster_mask = (label_map == cluster_id).astype(np.uint8)
        
        # Label connected components in the mask
        labeled_array, num_features = label(cluster_mask, structure=np.ones((3, 3)))
        
        # Process each connected component
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id)
            region_size = np.sum(region_mask)
            
            # Skip small regions
            if region_size < min_region_size:
                continue
            
            # Get pixels in this connected region
            region_pixels = image[region_mask].astype(np.float32)
            if len(region_pixels) == 0:
                continue
            
            # Compute mean color for the region
            region_mean = np.mean(region_pixels, axis=0)
            
            # Convert to HSV and apply random shift
            region_mean_hsv = rgb_to_hsv(region_mean[np.newaxis, :])[0]
            np.random.seed(42 + cluster_id + region_id)  # Unique seed for reproducibility
            shifts = np.random.uniform(-0.05, 0.05, size=3)  # Small shifts: ±5%
            region_mean_hsv += shifts * np.array([10.0, 0.1, 0.1])  # Scale: ±10° hue, ±0.05 sat/val
            region_mean_hsv[0] = np.clip(region_mean_hsv[0], 0, 360)  # Hue in [0, 360]
            region_mean_hsv[1] = np.clip(region_mean_hsv[1], 0, 1)    # Saturation in [0, 1]
            region_mean_hsv[2] = np.clip(region_mean_hsv[2], 0, 1)    # Value in [0, 1]
            
            # Convert back to RGB
            region_color = hsv_to_rgb(region_mean_hsv[np.newaxis, :])[0]
            region_color = np.clip(region_color, 0, 255).astype(np.uint8)
            
            # Assign the new color to the connected region
            output_image[region_mask] = region_color
    
    return output_image

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Blend similar and connected colors in an image with random HSV shift.")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to save the output image file")
    parser.add_argument("--tolerance", type=float, default=10.0, 
                        help="Tolerance for color similarity (default: 10.0)")
    parser.add_argument("--min-region-size", type=int, default=50,
                        help="Minimum number of pixels in a connected region (default: 50)")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load input image
        img = Image.open(args.input).convert("RGB")
        img_array = np.array(img)
        
        # Process image
        result_array = blend_colors(img_array, args.tolerance, args.min_region_size)
        
        # Save output image
        result_img = Image.fromarray(result_array)
        result_img.save(args.output)
        print(f"Processed image saved to {args.output}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
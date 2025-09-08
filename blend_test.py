import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import label, mean as ndi_mean
from scipy.spatial import cKDTree
from PIL import Image
import argparse

# Vectorized color conversions
def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB->[H(0..360), S(0..1), V(0..1)].
    rgb: (..., 3) in [0,255]
    """
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    nonzero = delta > 1e-8

    # r is max
    mask = nonzero & (maxc == r)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6
    # g is max
    mask = nonzero & (maxc == g)
    h[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2
    # b is max
    mask = nonzero & (maxc == b)
    h[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4

    h = h * 60.0  # degrees
    h[~nonzero] = 0.0

    # Saturation
    s = np.zeros_like(maxc)
    nonzero_max = maxc > 1e-8
    s[nonzero_max] = delta[nonzero_max] / maxc[nonzero_max]

    v = maxc
    hsv = np.stack([h, s, v], axis=-1)
    return hsv

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV->[0..255] RGB.
    hsv: (...,3) with H in [0,360], S,V in [0,1]
    """
    h = hsv[..., 0] / 60.0  # sector
    s = hsv[..., 1]
    v = hsv[..., 2]

    c = v * s
    x = c * (1 - np.abs((h % 2) - 1))
    m = v - c

    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)

    seg0 = (0 <= h) & (h < 1)
    seg1 = (1 <= h) & (h < 2)
    seg2 = (2 <= h) & (h < 3)
    seg3 = (3 <= h) & (h < 4)
    seg4 = (4 <= h) & (h < 5)
    seg5 = (5 <= h) & (h < 6)

    rp[seg0] = c[seg0]; gp[seg0] = x[seg0]; bp[seg0] = 0
    rp[seg1] = x[seg1]; gp[seg1] = c[seg1]; bp[seg1] = 0
    rp[seg2] = 0;        gp[seg2] = c[seg2]; bp[seg2] = x[seg2]
    rp[seg3] = 0;        gp[seg3] = x[seg3]; bp[seg3] = c[seg3]
    rp[seg4] = x[seg4]; gp[seg4] = 0;        bp[seg4] = c[seg4]
    rp[seg5] = c[seg5]; gp[seg5] = 0;        bp[seg5] = x[seg5]

    r = (rp + m)
    g = (gp + m)
    b = (bp + m)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

# Main blending pipeline
def blend_colors(image: np.ndarray, tolerance: float = 10.0, min_region_size: int = 50,
                 max_kmeans_samples: int = 100000) -> np.ndarray:

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array with uint8 dtype (H, W, C)")

    height, width, channels = image.shape
    assert channels == 3

    img_f = image.astype(np.float32)
    pixels = img_f.reshape(-1, 3)
    n_pixels = pixels.shape[0]

    # Determine number of clusters similar to original
    num_clusters = max(1, int(256 / tolerance))

    # Subsample pixels for k-means (faster)
    rng = np.random.default_rng(seed=12345)
    if n_pixels > max_kmeans_samples:
        sample_idx = rng.choice(n_pixels, size=max_kmeans_samples, replace=False)
    else:
        sample_idx = np.arange(n_pixels)

    sample_data = pixels[sample_idx]

    # Run kmeans on the sample to get centroids
    centroids, _ = kmeans2(sample_data, num_clusters, minit='points')

    # Assign every pixel to the nearest centroid in chunks to limit memory
    labels_all = np.empty(n_pixels, dtype=np.int32)
    chunk = 1_000_000  # ~1M pixels per chunk; tune if needed
    for start in range(0, n_pixels, chunk):
        end = min(start + chunk, n_pixels)
        block = pixels[start:end]  # (M,3)
        # distances to centroids (M, K)
        # compute squared distances
        # (a-b)^2 = a^2 + b^2 - 2ab; we do it explicitly to reduce temporaries
        a2 = np.sum(block * block, axis=1)[:, None]  # (M,1)
        b2 = np.sum(centroids * centroids, axis=1)[None, :]  # (1,K)
        ab = block.dot(centroids.T)  # (M,K)
        d2 = a2 + b2 - 2 * ab
        labels_all[start:end] = np.argmin(d2, axis=1)

    label_map = labels_all.reshape(height, width)

    output_image = image.copy()

    # Pre-allocate structure for connected component labeling
    structure = np.ones((3, 3), dtype=np.int8)

    # Iterate clusters
    for cluster_id in range(num_clusters):
        cluster_mask = (label_map == cluster_id).astype(np.uint8)
        if cluster_mask.sum() == 0:
            continue

        labeled_array, num_features = label(cluster_mask, structure=structure)

        if num_features == 0:
            continue

        # Use bincount to get sizes per feature (fast)
        counts = np.bincount(labeled_array.ravel())
        # index 0 is background; features are 1..num_features
        # find feature ids that meet min_region_size
        valid_ids = np.nonzero(counts >= min_region_size)[0]
        # drop background id=0
        valid_ids = valid_ids[valid_ids != 0]
        if valid_ids.size == 0:
            continue

        # Compute means per channel for the selected region ids using scipy.ndimage.mean
        # ndi_mean returns a list of means in the same order as 'index'
        idx_list = valid_ids.tolist()
        means_r = ndi_mean(img_f[..., 0], labels=labeled_array, index=idx_list)
        means_g = ndi_mean(img_f[..., 1], labels=labeled_array, index=idx_list)
        means_b = ndi_mean(img_f[..., 2], labels=labeled_array, index=idx_list)

        # Stack into (N_regions, 3)
        region_means = np.stack([means_r, means_g, means_b], axis=-1)  # still float (0..255)

        # Convert region means to HSV (vectorized)
        region_mean_hsv = rgb_to_hsv(region_means[np.newaxis, :, :].reshape(-1, 3))  # shape (N,3)
        # Apply deterministic small random shifts per region
        new_colors_rgb = np.empty_like(region_means, dtype=np.uint8)
        for i, region_label in enumerate(idx_list):
            # deterministic seed per cluster+region to replicate original's deterministic randomness
            seed_val = 42 + cluster_id + int(region_label)
            rng_region = np.random.default_rng(seed_val)
            shifts = rng_region.uniform(-0.05, 0.05, size=3)  # ±5% as before
            # original scaled shift: shifts * [10.0, 0.1, 0.1]
            hsv = region_mean_hsv[i].copy()
            hsv += shifts * np.array([10.0, 0.1, 0.1])
            hsv[0] = np.clip(hsv[0], 0, 360)
            hsv[1] = np.clip(hsv[1], 0, 1)
            hsv[2] = np.clip(hsv[2], 0, 1)
            rgb_new = hsv_to_rgb(hsv[np.newaxis, :])[0]
            new_colors_rgb[i] = rgb_new

            # Assign color to region in the output image
            mask = (labeled_array == int(region_label))
            output_image[mask] = rgb_new

    # Island of pixel absorbtion. To curb those ugly pixel islands.
    # mask of pixels that were changed by the region coloring pass
    changed_mask = np.any(output_image != image, axis=2)

    # If there are any unchanged pixels and at least one changed pixel, fix islands
    if not np.all(changed_mask) and changed_mask.any():
        changed_coords = np.column_stack(np.nonzero(changed_mask))  # (M,2) rows,cols
        changed_colors = output_image[changed_mask]  # (M,3)

        unchanged_coords = np.column_stack(np.nonzero(~changed_mask))  # (U,2)

        if changed_coords.shape[0] > 0 and unchanged_coords.shape[0] > 0:
            # cKDTree is fast; query nearest changed pixel for each unchanged pixel
            tree = cKDTree(changed_coords)
            _, idxs = tree.query(unchanged_coords, k=1)
            nearest_colors = changed_colors[idxs]

            # Assign nearest changed color to each originally-unchanged pixel
            output_image[~changed_mask] = nearest_colors

    return output_image

def main():
    parser = argparse.ArgumentParser(description="Blend similar and connected colors in an image with random HSV shift.")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to save the output image file")
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="Tolerance for color similarity (default: 10.0)")
    parser.add_argument("--min-region-size", type=int, default=50,
                        help="Minimum number of pixels in a connected region (default: 50)")
    args = parser.parse_args()

    try:
        img = Image.open(args.input).convert("RGB")
        img_array = np.array(img)
        result_array = blend_colors(img_array, args.tolerance, args.min_region_size)
        result_img = Image.fromarray(result_array)
        result_img.save(args.output)
        print(f"Processed image saved to {args.output}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

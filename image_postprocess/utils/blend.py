import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import distance_transform_edt, label
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Main blending pipeline

def blend_colors(image: np.ndarray, tolerance: float = 10.0, min_region_size: int = 50,
                          max_kmeans_samples: int = 100000, n_jobs: int | None = None) -> np.ndarray:
    """
    Blend large color regions after edge-preserving smoothing in LAB space.
    n_jobs: number of worker threads (None -> os.cpu_count()).
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array with uint8 dtype (H, W, C)")

    height, width, channels = image.shape
    assert channels == 3

    smoothed_image = cv2.bilateralFilter(image, 9, 75, 75)
    lab_image = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2LAB)

    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    n_pixels = pixels.shape[0]

    num_clusters = max(1, int(256 / tolerance))

    # Subsample for kmeans
    rng = np.random.default_rng(seed=12345)
    if n_pixels > max_kmeans_samples:
        sample_idx = rng.choice(n_pixels, size=max_kmeans_samples, replace=False)
    else:
        sample_idx = np.arange(n_pixels)
    sample_data = pixels[sample_idx]

    centroids, _ = kmeans2(sample_data, num_clusters, minit='points')
    cluster_colors = cv2.cvtColor(
        np.clip(centroids, 0, 255).astype(np.uint8)[np.newaxis, :, :],
        cv2.COLOR_LAB2RGB,
    )[0]

    # Assign every pixel to nearest centroid in chunks (same as original)
    labels_all = np.empty(n_pixels, dtype=np.int32)
    chunk = 1_000_000
    for start in range(0, n_pixels, chunk):
        end = min(start + chunk, n_pixels)
        block = pixels[start:end]  # (M,3)
        a2 = np.sum(block * block, axis=1)[:, None]
        b2 = np.sum(centroids * centroids, axis=1)[None, :]
        ab = block.dot(centroids.T)
        d2 = a2 + b2 - 2 * ab
        labels_all[start:end] = np.argmin(d2, axis=1)

    label_map = labels_all.reshape(height, width)
    output_image = image.copy()

    structure = np.ones((3, 3), dtype=np.int8)

    # Worker for a single cluster (runs in thread)
    def process_cluster(cluster_id: int):
        cluster_mask = (label_map == cluster_id).astype(np.uint8)
        if cluster_mask.sum() == 0:
            return 0  # nothing done

        labeled_array, num_features = label(cluster_mask, structure=structure)
        if num_features == 0:
            return 0

        counts = np.bincount(labeled_array.ravel())
        valid_ids = np.nonzero(counts >= min_region_size)[0]
        valid_ids = valid_ids[valid_ids != 0]
        if valid_ids.size == 0:
            return 0

        idx_list = valid_ids.tolist()
        cluster_color = cluster_colors[cluster_id]
        for region_label in idx_list:
            mask = (labeled_array == int(region_label))
            output_image[mask] = cluster_color

        return 1  # done something

    # Run cluster processing in thread pool
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, int(n_jobs))

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(process_cluster, cid) for cid in range(num_clusters)]
        # optional: iterate to ensure completion
        for _ in as_completed(futures):
            pass

    # Fill untouched islands from their nearest blended pixel.
    changed_mask = np.any(output_image != image, axis=2)
    if not np.all(changed_mask) and changed_mask.any():
        _, indices = distance_transform_edt(~changed_mask, return_indices=True)
        nearest_y = indices[0][~changed_mask]
        nearest_x = indices[1][~changed_mask]
        output_image[~changed_mask] = output_image[nearest_y, nearest_x]

    return output_image

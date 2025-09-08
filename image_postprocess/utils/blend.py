import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import label, mean as ndi_mean
from scipy.spatial import cKDTree
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                          max_kmeans_samples: int = 100000, n_jobs: int | None = None) -> np.ndarray:
    """
    Parallelized version of blend_colors.
    n_jobs: number of worker threads (None -> os.cpu_count()).
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array with uint8 dtype (H, W, C)")

    height, width, channels = image.shape
    assert channels == 3

    img_f = image.astype(np.float32)
    pixels = img_f.reshape(-1, 3)
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
        means_r = ndi_mean(img_f[..., 0], labels=labeled_array, index=idx_list)
        means_g = ndi_mean(img_f[..., 1], labels=labeled_array, index=idx_list)
        means_b = ndi_mean(img_f[..., 2], labels=labeled_array, index=idx_list)
        region_means = np.stack([means_r, means_g, means_b], axis=-1)  # float 0..255

        # convert region means to HSV and generate new colors per region
        region_mean_hsv = rgb_to_hsv(region_means[np.newaxis, :, :].reshape(-1, 3))
        # iterate regions (small loop per region; still OK)
        for i, region_label in enumerate(idx_list):
            seed_val = 42 + cluster_id + int(region_label)
            rng_region = np.random.default_rng(seed_val)
            shifts = rng_region.uniform(-0.05, 0.05, size=3)
            hsv = region_mean_hsv[i].copy()
            hsv += shifts * np.array([10.0, 0.1, 0.1])
            hsv[0] = np.clip(hsv[0], 0, 360)
            hsv[1] = np.clip(hsv[1], 0, 1)
            hsv[2] = np.clip(hsv[2], 0, 1)
            rgb_new = hsv_to_rgb(hsv[np.newaxis, :])[0]

            mask = (labeled_array == int(region_label))
            # assign directly into shared output_image; clusters don't overlap so this is safe
            output_image[mask] = rgb_new

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

    # Island absorbtion (parallelize KD-tree queries by chunking queries)
    changed_mask = np.any(output_image != image, axis=2)
    if not np.all(changed_mask) and changed_mask.any():
        changed_coords = np.column_stack(np.nonzero(changed_mask))  # (M,2)
        changed_colors = output_image[changed_mask]  # (M,3)
        unchanged_coords = np.column_stack(np.nonzero(~changed_mask))  # (U,2)

        if changed_coords.shape[0] > 0 and unchanged_coords.shape[0] > 0:
            tree = cKDTree(changed_coords)

            # We'll chunk the unchanged coords and parallel query
            def query_chunk(start_end):
                s, e = start_end
                sub = unchanged_coords[s:e]
                _, idxs = tree.query(sub, k=1)
                return (s, e, idxs)

            # prepare ranges
            U = unchanged_coords.shape[0]
            qchunk = max(1_000, U // (n_jobs * 4) + 1)
            ranges = [(i, min(i + qchunk, U)) for i in range(0, U, qchunk)]

            nearest_colors = np.empty((U, 3), dtype=np.uint8)
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                futures = [ex.submit(query_chunk, r) for r in ranges]
                for fut in as_completed(futures):
                    s, e, idxs = fut.result()
                    nearest_colors[s:e] = changed_colors[idxs]

            # assign back
            # flatten indexing: map (r,c) to flat index
            flat_idx = unchanged_coords[:, 0] * width + unchanged_coords[:, 1]
            out_flat = output_image.reshape(-1, 3)
            out_flat[flat_idx] = nearest_colors

    return output_image

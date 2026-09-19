import torch
import torch.optim as optim
import lpips
import torchvision.transforms as transforms
import numpy as np
import sys

def attack_non_semantic(img_arr: np.ndarray,
            iterations: int = 500,
            learning_rate: float = 3e-4,
            t_lpips: float = 4e-2,
            t_l2: float = 3e-5,
            c_lpips: float = 1e-2,
            c_l2: float = 0.6,
            grad_clip_value: float = 0.05,
            adaptive_c_lpips: bool = True,
            c_lpips_max: float = 1.0,
            c_lpips_min: float = 1e-4,
            search_interval: int = 50,
            t_l2_override: float | None = None
    ) -> np.ndarray:
    """
    Implements the non-semantic attack from the UnMarker paper (Section 5.1) using numpy input/output.

    Args:
        img_arr: Input image as a numpy array (H, W, 3) in range [0, 255].
        iterations: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        t_lpips: Threshold for LPIPS loss.
        t_l2: Per-pixel RMS budget for the L2 constraint. The effective global
            threshold is t_l2 * sqrt(numel), so it scales with resolution
            instead of being an unusable fixed global norm.
        t_l2_override: Explicit global L2 threshold. When provided, it replaces
            the resolution-scaled threshold derived from t_l2.
        c_lpips: LPIPS loss weight constant (used only when adaptive_c_lpips is False).
        c_l2: L2 loss weight constant (fixed coefficient per Section 5.1).
        grad_clip_value: Gradient clipping value.
        adaptive_c_lpips: If True, c_lpips is determined by a binary search that
            starts large, decreases on success, and amplifies on failure, as
            described in Section 5.1 of the UnMarker paper. The search brackets
            the coefficient on a multiplicative (geometric) scale, and each
            trial restarts the perturbation and optimizer state so the success
            test reflects the current coefficient.
        c_lpips_max: Upper bound and starting value for the adaptive c_lpips search.
        c_lpips_min: Lower bound for the adaptive c_lpips search.
        search_interval: Number of optimization steps per adaptive search trial.

    Returns:
        Attacked image as a numpy array (H, W, 3) in range [0, 255]. When at
        least one feasible iterate was found, the feasible iterate with the
        largest DFL is returned; otherwise the last iterate is returned.
    """
    config = {
        'iterations': iterations,
        'learning_rate': learning_rate,
        't_lpips': t_lpips,
        't_l2': t_l2,
        't_l2_override': t_l2_override,
        'c_lpips': c_lpips,
        'c_l2': c_l2,
        'grad_clip_value': grad_clip_value,
        'adaptive_c_lpips': adaptive_c_lpips,
        'c_lpips_max': c_lpips_max,
        'c_lpips_min': c_lpips_min,
        'search_interval': max(1, int(search_interval))
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-initialized so the finally-block cleanup never raises
    # UnboundLocalError and masks the original exception when a failure
    # happens before/during model construction.
    img_tensor = None
    delta = None
    optimizer = None
    lpips_model = None
    ref_feats = None
    lpips_loss = None
    img_fft = None
    has_best = None
    best_dfl = None
    best_delta_eff = None
    x_fin = None
    final_x_nw = None

    try:
        # Preprocess: Convert numpy array to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img_arr).unsqueeze(0).to(device)

        # Initialize perturbation
        delta = (torch.randn_like(img_tensor) * 1e-5).requires_grad_(True).to(device)

        # Setup optimizer and LPIPS model
        optimizer = optim.Adam([delta], lr=config['learning_rate'])
        _, _, img_h, img_w = img_tensor.shape
        lpips_net = 'vgg' if (img_h >= 256 and img_w >= 256) else 'alex'
        lpips_model = lpips.LPIPS(net=lpips_net).to(device)

        # Cache fixed-reference LPIPS trunk outputs once: img_tensor never changes,
        # so only the x-side trunk needs to run per call.
        with torch.no_grad():
            ref_feats = [lpips.normalize_tensor(f).detach()
                         for f in lpips_model.net.forward(lpips_model.scaling_layer(img_tensor))]

        def lpips_loss(x):
            x_feats = lpips_model.net.forward(lpips_model.scaling_layer(x))
            val = None
            for k in range(lpips_model.L):
                diff = (lpips.normalize_tensor(x_feats[k]) - ref_feats[k]) ** 2
                layer = lpips.spatial_average(lpips_model.lins[k](diff), keepdim=True)
                val = layer if val is None else val + layer
            return val.mean()

        # Precompute FFT of input image
        img_fft = torch.fft.fft2(img_tensor)

        numel = img_tensor.numel()
        if config['t_l2_override'] is not None:
            t_l2_eff = float(config['t_l2_override'])
        else:
            t_l2_eff = config['t_l2'] * (numel ** 0.5)

        if config['adaptive_c_lpips']:
            c_lo = config['c_lpips_min']
            c_hi = config['c_lpips_max']
            current_c = config['c_lpips_max']
        else:
            current_c = config['c_lpips']

        has_best = torch.zeros((), dtype=torch.bool, device=device)
        best_dfl = torch.zeros((), dtype=torch.float32, device=device)
        best_delta_eff = torch.zeros_like(img_tensor)

        def eval_feasible() -> bool:
            with torch.no_grad():
                x_eval = torch.clamp(img_tensor + delta, -1, 1)
                lp = lpips_loss(x_eval)
                l2 = torch.linalg.norm(x_eval - img_tensor)
                lp_val, l2_val = torch.stack([lp, l2]).cpu().tolist()
            return lp_val <= config['t_lpips'] and l2_val <= t_l2_eff

        # Optimization loop
        for i in range(config['iterations']):
            optimizer.zero_grad()

            # Perturbed image
            x_nw = torch.clamp(img_tensor + delta, -1, 1)

            # Effective perturbation after clamping, so the L2 objective matches
            # the perturbation actually present in the output
            delta_eff = x_nw - img_tensor

            # Spectral Loss (DFL)
            x_nw_fft = torch.fft.fft2(x_nw)
            loss_dfl = -torch.abs(x_nw_fft - img_fft).sum()

            # Perceptual Loss (LPIPS)
            loss_lpips = lpips_loss(x_nw)

            # Geometric Loss (L2 Norm)
            loss_l2 = torch.linalg.norm(delta_eff)

            # Combine losses
            lpips_penalty = current_c * torch.relu(loss_lpips - config['t_lpips'])
            l2_penalty = config['c_l2'] * torch.relu(loss_l2 - t_l2_eff)
            total_loss = loss_dfl + lpips_penalty + l2_penalty

            # Backpropagation
            total_loss.backward()

            # Gradient clipping
            if delta.grad is not None:
                delta.grad.data.clamp_(-config['grad_clip_value'], config['grad_clip_value'])

            optimizer.step()

            with torch.no_grad():
                lp_val = loss_lpips.detach()
                l2_val = loss_l2.detach()
                d_val = loss_dfl.detach()
                feasible = (lp_val <= config['t_lpips']) & (l2_val <= t_l2_eff)
                better = feasible & (~has_best | (d_val < best_dfl))
                has_best = has_best | feasible
                best_dfl = torch.where(better, d_val.to(best_dfl.dtype), best_dfl)
                best_delta_eff = torch.where(better, delta_eff.detach(), best_delta_eff)

            # Adaptive c_lpips update at the end of each search trial
            if (config['adaptive_c_lpips']
                    and (i + 1) % config['search_interval'] == 0
                    and (i + 1) < config['iterations']):
                if eval_feasible():
                    c_hi = current_c
                else:
                    c_lo = current_c
                current_c = (c_lo * c_hi) ** 0.5
                delta = (torch.randn_like(img_tensor) * 1e-5).requires_grad_(True).to(device)
                optimizer = optim.Adam([delta], lr=config['learning_rate'])

            n = config['iterations']
            frac = (i + 1) / n
            filled = int(frac * 30)
            sys.stdout.write('\r[' + '#' * filled + '-' * (30 - filled) + f'] {frac * 100:5.1f}% {i + 1}/{n}')
            sys.stdout.flush()
            if i + 1 == n:
                sys.stdout.write('\n')

        with torch.no_grad():
            x_fin = torch.clamp(img_tensor + delta, -1, 1)
            lp_fin = lpips_loss(x_fin)
            l2_fin = torch.linalg.norm(x_fin - img_tensor)
            feasible_fin = (lp_fin <= config['t_lpips']) & (l2_fin <= t_l2_eff)
            d_fin = -torch.abs(torch.fft.fft2(x_fin) - img_fft).sum()
            better_fin = feasible_fin & (~has_best | (d_fin < best_dfl))
            has_best = has_best | feasible_fin
            best_dfl = torch.where(better_fin, d_fin.to(best_dfl.dtype), best_dfl)
            best_delta_eff = torch.where(better_fin, x_fin - img_tensor, best_delta_eff)

        # Postprocess: Convert back to numpy array
        final_x_nw = torch.where(has_best, img_tensor + best_delta_eff, x_fin)
        final_x_nw = torch.clamp(final_x_nw, -1, 1)
        final_x_nw = final_x_nw.squeeze(0).cpu().detach()
        final_x_nw = (final_x_nw + 1) / 2  # Denormalize to [0, 1]
        final_x_nw = final_x_nw.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
        final_x_nw = final_x_nw.clamp(0, 1) * 255  # Scale to [0, 255]
        result = final_x_nw.numpy().astype(np.uint8)
    finally:
        # Drop every GPU-owning reference (latest live bindings, including
        # rebinding during the adaptive search) plus the closures whose cells
        # capture the model and cached features, so neither frame locals nor
        # closure cells retain them past this point.
        del img_tensor, delta, optimizer, lpips_model, ref_feats, lpips_loss
        del img_fft, has_best, best_dfl, best_delta_eff, x_fin, final_x_nw
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return result

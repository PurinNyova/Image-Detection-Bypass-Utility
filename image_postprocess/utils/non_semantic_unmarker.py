import torch
import torch.optim as optim
import lpips
import torchvision.transforms as transforms
import numpy as np

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

    best_delta_eff = None
    best_dfl = None

    def eval_feasible() -> bool:
        with torch.no_grad():
            x_eval = torch.clamp(img_tensor + delta, -1, 1)
            lp = lpips_model(x_eval, img_tensor).mean().item()
            l2 = torch.linalg.norm(x_eval - img_tensor).item()
        return lp <= config['t_lpips'] and l2 <= t_l2_eff

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
        loss_lpips = lpips_model(x_nw, img_tensor).mean()

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

        lp_val = float(loss_lpips.detach())
        l2_val = float(loss_l2.detach())
        if lp_val <= config['t_lpips'] and l2_val <= t_l2_eff:
            d_val = float(loss_dfl.detach())
            if best_dfl is None or d_val < best_dfl:
                best_dfl = d_val
                best_delta_eff = delta_eff.detach().clone()

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

    with torch.no_grad():
        x_fin = torch.clamp(img_tensor + delta, -1, 1)
        lp_fin = lpips_model(x_fin, img_tensor).mean().item()
        l2_fin = torch.linalg.norm(x_fin - img_tensor).item()
        if lp_fin <= config['t_lpips'] and l2_fin <= t_l2_eff:
            d_fin = float(-torch.abs(torch.fft.fft2(x_fin) - img_fft).sum())
            if best_dfl is None or d_fin < best_dfl:
                best_dfl = d_fin
                best_delta_eff = x_fin - img_tensor

    # Postprocess: Convert back to numpy array
    if best_delta_eff is not None:
        final_x_nw = img_tensor + best_delta_eff
    else:
        final_x_nw = torch.clamp(img_tensor + delta, -1, 1)
    final_x_nw = torch.clamp(final_x_nw, -1, 1)
    final_x_nw = final_x_nw.squeeze(0).cpu().detach()
    final_x_nw = (final_x_nw + 1) / 2  # Denormalize to [0, 1]
    final_x_nw = final_x_nw.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
    final_x_nw = final_x_nw.clamp(0, 1) * 255  # Scale to [0, 255]
    result = final_x_nw.numpy().astype(np.uint8)

    return result

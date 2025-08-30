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
            grad_clip_value: float = 0.05
    ) -> np.ndarray:
    """
    Implements the non-semantic attack from the UnMarker paper using numpy input/output.
    
    Args:
        img_arr: Input image as a numpy array (H, W, 3) in range [0, 255].
        iterations: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        t_lpips: Threshold for LPIPS loss.
        t_l2: Threshold for L2 loss.
        c_lpips: LPIPS loss weight constant.
        c_l2: L2 loss weight constant.
        grad_clip_value: Gradient clipping value.
    
    Returns:
        Attacked image as a numpy array (H, W, 3) in range [0, 255].
    """
    # Build configuration dictionary from parameters
    config = {
        'iterations': iterations,
        'learning_rate': learning_rate,
        't_lpips': t_lpips,
        't_l2': t_l2,
        'c_lpips': c_lpips,
        'c_l2': c_l2,
        'grad_clip_value': grad_clip_value
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
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Precompute FFT of input image
    img_fft = torch.fft.fft2(img_tensor)

    # Optimization loop
    for i in range(config['iterations']):
        optimizer.zero_grad()

        # Perturbed image
        x_nw = img_tensor + delta
        x_nw = torch.clamp(x_nw, -1, 1)

        # Spectral Loss (DFL)
        x_nw_fft = torch.fft.fft2(x_nw)
        loss_dfl = -torch.abs(x_nw_fft - img_fft).sum()

        # Perceptual Loss (LPIPS)
        loss_lpips = lpips_model(x_nw, img_tensor).mean()

        # Geometric Loss (L2 Norm)
        loss_l2 = torch.linalg.norm(delta)

        # Combine losses
        lpips_penalty = config['c_lpips'] * torch.relu(loss_lpips - config['t_lpips'])
        l2_penalty = config['c_l2'] * torch.relu(loss_l2 - config['t_l2'])
        total_loss = loss_dfl + lpips_penalty + l2_penalty

        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        if delta.grad is not None:
            delta.grad.data.clamp_(-config['grad_clip_value'], config['grad_clip_value'])

        optimizer.step()

    # Postprocess: Convert back to numpy array
    final_x_nw = torch.clamp(img_tensor + delta, -1, 1)
    final_x_nw = final_x_nw.squeeze(0).cpu().detach()
    final_x_nw = (final_x_nw + 1) / 2  # Denormalize to [0, 1]
    final_x_nw = final_x_nw.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
    final_x_nw = final_x_nw.clamp(0, 1) * 255  # Scale to [0, 255]
    result = final_x_nw.numpy().astype(np.uint8)

    return result
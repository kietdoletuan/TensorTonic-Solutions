import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.detach().clone().requires_grad_(True)
    last_grad = None

    for i, (x, y) in enumerate(micro_batches):
        pred = torch.dot(w, x)
        loss = (pred - y)**2

        loss.backward()

        if (i + 1) % accum_steps == 0:
            last_grad = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr * last_grad
            w.grad = None
            
    return w, last_grad

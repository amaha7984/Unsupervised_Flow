import torch
from torchdiffeq import odeint
from torchvision.utils import save_image


@torch.no_grad()
def translate_tensor_ode(
    model,
    x0,
    steps=50,
    method="dopri5",
    tol=1e-5,
):

    model.eval()
    B = x0.shape[0]

    def f(t, x):
        # t is scalar from odeint; expanding to batch to match training
        t_batch = torch.full((B,), t, device=x.device, dtype=x.dtype)
        return model(t_batch, x)

    if method == "euler":
        t_grid = torch.linspace(0, 1, steps + 1, device=x0.device, dtype=x0.dtype)
        x = x0
        dt = 1.0 / float(steps)
        for k in range(steps):
            t = t_grid[k]
            dx = f(t, x)
            x = x + dt * dx
        return x

    t_span = torch.linspace(0, 1, 2, device=x0.device, dtype=x0.dtype)
    xT = odeint(f, x0, t_span, rtol=tol, atol=tol, method=method)[-1]
    return xT


@torch.no_grad()
def translate_and_save_grid(
    model,
    vae,
    xA,
    out_path,
    ode_steps=50,
    ode_method="dopri5",
    ode_tol=1e-5,
):
    """
    Saves a grid: first row = inputs (A), second row = translated outputs (fake B).

    - If vae is NOT None: run latent pipeline
        xA -> encode -> zA -> ODE -> zB_hat -> decode -> xB_hat
    - If vae is None: run pixel pipeline
        xA -> ODE -> xB_hat

    xA is in [-1,1].
    """
    model.eval()

    if vae is not None:
        vae.eval()
        zA = vae.encode(xA).latent_dist.mean
        zB_hat = translate_tensor_ode(model, zA, steps=ode_steps, method=ode_method, tol=ode_tol)
        xB_hat = vae.decode(zB_hat).sample
    else:
        xB_hat = translate_tensor_ode(model, xA, steps=ode_steps, method=ode_method, tol=ode_tol)

    xA_vis = (xA.clamp(-1, 1) + 1) / 2.0
    xB_vis = (xB_hat.clamp(-1, 1) + 1) / 2.0

    grid = torch.cat([xA_vis, xB_vis], dim=0)
    save_image(grid, out_path, nrow=xA.shape[0])

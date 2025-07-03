import torch
from itertools import permutations
from create_data import generate_quadratic_pc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  # Highest precision

def centralize(X: torch.Tensor) -> torch.Tensor:
    """Center the point cloud by subtracting its mean."""
    return X - X.mean(dim=0, keepdim=True)

def compute_f(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    f(X, Y) = min_π ∑_{i,j} |‖x_i - x_j‖ - ‖y_π(i) - y_π(j)‖|

    Args:
        X, Y: (n, 2) point clouds (centered)

    Returns:
        Minimum distortion over permutations
    """
    n = X.size(0)
    min_val = torch.tensor(float('inf'), dtype=dtype, device=X.device)

    for perm in permutations(range(n)):
        Y_perm = Y[list(perm)]
        diff = torch.tensor(0.0, dtype=dtype, device=X.device)
        for i in range(n):
            for j in range(n):
                dX = torch.norm(X[i] - X[j])
                dY = torch.norm(Y_perm[i] - Y_perm[j])
                diff += torch.abs(dX - dY)
        min_val = min(min_val, diff)

    return min_val.item()

def compute_g(X: torch.Tensor, Y: torch.Tensor, m: int = 100) -> float:
    """
    g(X, Y) = min_{π, R} ∑_i ‖y_π(i) - x_i R‖

    Args:
        X, Y: (n, 2) point clouds (centered)
        m: number of angles to sample

    Returns:
        Minimum alignment error
    """
    n = X.size(0)
    min_val = torch.tensor(float('inf'), dtype=dtype, device=X.device)

    angles = torch.linspace(0, 2 * torch.pi, m, dtype=dtype, device=X.device)

    for perm in permutations(range(n)):
        Y_perm = Y[list(perm)]
        for angle in angles:
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            R = torch.stack([
                torch.stack([cos_a, -sin_a]),
                torch.stack([sin_a,  cos_a])
            ])
            X_rot = X @ R.T
            diff = torch.norm(Y_perm - X_rot, dim=1).sum()
            min_val = min(min_val, diff)

    return min_val.item()

def compute_f_over_g_with_random_X_Y(m: int = 100, eps: float = 1e-3):
    """
    Generates perturbed point clouds and computes f(X,Y) / g(X,Y)

    Returns:
        f_value, g_value
    """
    data_X, data_Y = generate_quadratic_pc(eps)
    X = data_X.pos.to(dtype).to(device)
    Y = data_Y.pos.to(dtype).to(device)

    X = centralize(X)
    Y = centralize(Y)

    f_val = compute_f(X, Y)
    g_val = compute_g(X, Y, m=m)

    return f_val, g_val

if __name__ == "__main__":
    f_val, g_val = compute_f_over_g_with_random_X_Y(m=200, eps=1e-6)
    ratio = f_val / g_val if g_val != 0 else float('inf')
    print(f"f = {f_val}, g = {g_val}, ratio = {ratio}")


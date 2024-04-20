import torch
import numpy as np
from typing import Callable
from utils.utils import *

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def sample_surface_eta_omega(param, sample_num, eta, omega):
    """

    :param param: a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4])
           index  0   1   2   3   4   5   6   7   8   9   10  11  12  13
    :param sample_num:
    :return:
    """
    def fexp(x, p):
        return torch.sign(x) * (torch.abs(x) ** p)

    x = param[0] * fexp(torch.cos(eta), param[3]) * fexp(torch.cos(omega), param[4])
    y = param[1] * fexp(torch.cos(eta), param[3]) * fexp(torch.sin(omega), param[4])
    z = param[2] * fexp(torch.sin(eta), param[3])
    points = torch.stack((x, y, z)).T
    return points

class PRNG:
    def __init__(self, seed: int):
        self.gen = np.random.Generator(np.random.PCG64(seed))
        self.dis = np.random.uniform

    def __call__(self) -> float:
        return self.dis()

class RecursionParams:
    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        theta_a: float,
        theta_b: float,
        n: int,
        offset: int
    ):
        self.A = a
        self.B = b
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.N = n
        self.offset = offset

def fexp(x: float, p: float) -> float:
    return np.copysign(np.power(np.abs(x), p), x)

def xy(theta: float, a1: float, a2: float, e: float) -> np.ndarray:
    out = np.zeros(2)
    out[0] = a1 * fexp(np.cos(theta), e)
    out[1] = a2 * fexp(np.sin(theta), e)
    return out

def distance(a: np.ndarray, b: np.ndarray) -> float:
    d1 = a[0] - b[0]
    d2 = a[1] - b[1]
    return np.sqrt(d1**2 + d2**2)

def sample_superellipse_divide_conquer(
    a1: float,
    a2: float,
    e: float,
    theta_a: float,
    theta_b: float,
    buffer_size: int
) -> np.ndarray:
    A = xy(theta_a, a1, a2, e)
    B = xy(theta_b, a1, a2, e)
    buffer = np.zeros(buffer_size)
    buffer[0] = theta_a
    stack = [RecursionParams(A, B, theta_a, theta_b, buffer_size-2, 1)]

    while stack:
        params = stack.pop()
        if params.N <= 0:
            continue

        theta = (params.theta_a + params.theta_b) / 2
        C = xy(theta, a1, a2, e)
        dA = distance(params.A, C)
        dB = distance(C, params.B)
        nA = int(round((dA / (dA + dB + 1e-6)) * (params.N - 1)))
        nB = params.N - nA - 1

        buffer[nA + params.offset] = theta

        stack.append(RecursionParams(params.A, C, params.theta_a, theta, nA, params.offset))
        stack.append(RecursionParams(C, params.B, theta, params.theta_b, nB, params.offset + nA + 1))

    buffer[-1] = theta_b
    return buffer

def sample_etas(
    rand: Callable[[], float],
    a1a2: float,
    e1: float,
    buffer: np.ndarray
) -> np.ndarray:
    smoothing = 0.001
    cdf = np.zeros_like(buffer)
    cdf[0] = smoothing
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i-1] + smoothing + a1a2 * fexp(np.cos(buffer[i]), e1)
    s = cdf[-1]
    cdf /= s

    etas = np.zeros_like(buffer)
    for i in range(len(etas)):
        pos = np.searchsorted(cdf, rand())
        etas[i] = buffer[pos]
    return etas

def sample_on_batch(
    shapes: np.ndarray,
    epsilons: np.ndarray,
    N: int,
    seed: int
):
    rand = PRNG(seed)

    buffer = sample_superellipse_divide_conquer(
        shapes[0],
        shapes[2],
        epsilons[0],
        np.pi/2, -np.pi/2,
        N
    )
    etas = sample_etas(
        rand,
        shapes[0]+shapes[1],
        epsilons[0],
        buffer
    )

    buffer = sample_superellipse_divide_conquer(
        shapes[0],
        shapes[1],
        epsilons[1],
        np.pi, -np.pi,
        N
    )
    omegas = buffer[np.random.randint(0, N, size=N)]

    return etas, omegas

def batch_sample(sp_param, sample_num):
    """
    sample points from superquadrics defined by sp_param
    :param sp_param: define superquadrics
    :param sample_num: num of points sampled from per superquadric
    :return: sample_points, B M sample_num 3
    """

    B, M, _ = sp_param.shape  # B: batch size, M: superpoint num per object

    sample_points = []
    for batch in sp_param:
        sample_points_b = []
        for param in batch:
            size = param[:3].detach().cpu().numpy()
            shape = param[3:5].detach().cpu().numpy()
            eta, omega = sample_on_batch(size, shape, sample_num, 0)
            eta[eta == 0] += 1e-6
            omega[omega == 0] += 1e-6
            eta = param.new_tensor(eta)
            omega = param.new_tensor(omega)
            points = sample_surface_eta_omega(param, sample_num, eta, omega)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
            y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
            z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))
            # deform
            fx = param[5] * z / param[2] + 1
            fy = param[6] * z / param[2] + 1
            fz = 1
            x = x * fx
            y = y * fy
            z = z * fz
            points = torch.stack([x, y, z]).T
            points = points.unsqueeze(0)
            sample_points_b.append(points)
        sample_points_b = torch.cat(sample_points_b, dim=0).unsqueeze(0)
        sample_points.append(sample_points_b)
    sample_points = torch.cat(sample_points, dim=0)
    return sample_points

def distance_point2superquadric(points, sq_points):
    """

    :param points: B N 3
    :param sq_points: B M L 3, L is num of points sampled from superquadrics
    :return:
    """
    # distance
    points = points.unsqueeze(3)        # B N M 1 3
    sq_points = sq_points.unsqueeze(1)  # B 1 M L 3
    distance = points - sq_points       # B N M L 3
    distance = torch.norm(distance, dim=-1)  # B N M L
    distance = distance.min(dim=-1)[0]  # B N M

    return distance

def distance_superquadric2point(points, sq_points, one_hot):
    """

    :param points: B N 3
    :param sq_points: B M L 3, L is num of points sampled from superquadrics
    :param one_hot: B M N
    :return:
    """
    _, _, L, _ = sq_points.shape

    # distance
    sq_points = sq_points.unsqueeze(3)  # B M L 1 3
    points = points.transpose(2, 1).unsqueeze(2)  # B M 1 N 3
    distance = points - sq_points  # B M L N 3
    distance = torch.norm(distance, dim=-1)  # B M L N
    distance = distance.transpose(2, 1)  # B L M N

    # one hot mask
    one_hot = one_hot.unsqueeze(1).expand(-1, L, -1, -1)  # B L M N
    distance[~one_hot.bool()] = float('inf')  # B L M N
    distance = distance.min(dim=-1)[0]  # B L M
    inf_nan_to_num(distance)

    return distance


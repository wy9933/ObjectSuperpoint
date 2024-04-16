import torch
import open3d as o3d
import numpy as np

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

def sample_surface(param, sample_num):
    x = (2 * torch.rand(sample_num) - 1) * param[0]
    y = (2 * torch.rand(sample_num) - 1) * param[1]
    xt = torch.pow(torch.abs(x / param[0]), 2 / param[4])
    yt = torch.pow(torch.abs(y / param[1]), 2 / param[4])
    zt = 1 - torch.pow(xt + yt, param[4] / param[3])
    ztt = torch.pow(torch.abs(zt), param[3] / 2) * torch.sign(zt)
    z = ztt * param[2]
    points_z = torch.stack((x, y, z)).T
    points_z = points_z[points_z[:, 2] >= 0]
    x1 = (2 * torch.rand(sample_num) - 1) * param[0]
    y1 = (2 * torch.rand(sample_num) - 1) * param[1]
    xt1 = torch.pow(torch.abs(x1 / param[0]), 2 / param[4])
    yt1 = torch.pow(torch.abs(y1 / param[1]), 2 / param[4])
    zt1 = 1 - torch.pow(xt1 + yt1, param[4] / param[3])
    ztt1 = torch.pow(torch.abs(zt1), param[3] / 2) * torch.sign(zt1)
    z1 = ztt1 * param[2]
    points_f = torch.stack((x1, y1, -z1)).T
    points_f = points_f[points_f[:, 2] <= 0]
    points = torch.cat((points_z, points_f), dim=0)
    return points

def test_gradient(a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4):
    # 测试梯度
    param = torch.FloatTensor([a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4])
    # index                    0   1   2   3   4   5   6   7   8   9   10  11  12  13
    param.requires_grad = True

    # 确定xy计算z的方式在superquadric表面采样
    sample_num = 100000
    points = sample_surface(param, sample_num)
    # gradient = torch.autograd.grad(torch.sum(points), param)
    # print("gradient after sample surface", gradient)

    # bending形变
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fx = param[5] * z / a3
    fx += 1
    fy = param[6] * z / a3
    fy += 1
    fz = 1
    x = x * fx
    y = y * fy
    z = z * fz
    points = torch.stack([x, y, z]).T
    # gradient = torch.autograd.grad(torch.sum(points), param)
    # print("gradient after bending", gradient)

    # 旋转
    # 这里是superquadric在旋转，R.T
    # 到时候计算损失的时候需要把shape中的点旋转到superquadric坐标系下
    # 旋转shape中点的时候使用的方法是torch.matmul(points, R)
    # 因为R旋转矩阵是个正交矩阵，其逆矩阵就等于其转置
    # 旋转superquadric用的是转置，那么旋转shape点就用其本身就好了
    R = quaternion_to_matrix(param[-4:])
    points = torch.matmul(points, R.T)
    # gradient = torch.autograd.grad(torch.sum(points), param)
    # print("gradient after rotate", gradient)

    # 平移
    # superquadric平移是加
    # shape点平移就是减
    T = param[7:10]
    points += T
    gradient = torch.autograd.grad(torch.sum(points), param)
    print("gradient after translate\n", gradient)

    # 并且需要注意的是：
    # 计算损失的时候，superquadric：采样->形变
    # shape点：逆平移->逆旋转

    points = points.detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # 大小
    a1, a2, a3 = 0.5, 0.5, 0.5
    # 形状
    e1, e2 = 1.25, 0.25
    # 弯曲
    K1, K2 = 0.3, 0.5
    # 平移
    t1, t2, t3 = 1, 1, 1
    # 旋转
    q1, q2, q3, q4 = 0.5, 0.5, 0.1, 0.6

    test_gradient(a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4)
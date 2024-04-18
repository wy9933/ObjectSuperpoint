import torch
import numpy as np

from easydict import EasyDict


class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]


def print_log(args, string):
    with open(args.log_file, 'a') as f:
        f.write(string + "\n")
    print(string)


def log_args_to_file(args, pre='args'):
    for key, val in args.__dict__.items():
        print_log(args, f'{pre}.{key} : {val}')


def log_config_to_file(args, config, pre='config'):
    for key, val in config.items():
        if isinstance(config[key], EasyDict):
            print_log(args, f'{pre}.{key} = easydict()')
            log_config_to_file(args, config[key], pre=pre + '.' + key)
            continue
        print_log(args, f'{pre}.{key} : {val}')


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


# do not use, it will cause nan in model param grad
def sample_surface(param, sample_num):
    x = 2 * torch.rand(sample_num, device=param.device) - 1
    y = 2 * torch.rand(sample_num, device=param.device) - 1
    xt = torch.pow(torch.abs(x), 2 / (param[4] + 1e-4))
    yt = torch.pow(torch.abs(y), 2 / (param[4] + 1e-4))
    z = 1 - torch.pow(xt + yt, param[4] / (param[3] + 1e-4))
    z = torch.pow(torch.abs(z), param[3] / 2) * torch.sign(z) * param[2]
    x *= param[0]
    y *= param[1]
    points_z = torch.stack((x, y, z)).T
    points_z = points_z[points_z[:, 2] >= 0]

    x = 2 * torch.rand(sample_num, device=param.device) - 1
    y = 2 * torch.rand(sample_num, device=param.device) - 1
    xt = torch.pow(torch.abs(x), 2 / (param[4] + 1e-4))
    yt = torch.pow(torch.abs(y), 2 / (param[4] + 1e-4))
    z = 1 - torch.pow(xt + yt, param[4] / (param[3] + 1e-4))
    z = torch.pow(torch.abs(z), param[3] / 2) * torch.sign(z) * param[2]
    x *= param[0]
    y *= param[1]
    points_f = torch.stack((x, y, -z)).T
    points_f = points_f[points_f[:, 2] <= 0]

    points = torch.cat((points_z, points_f), dim=0)
    return points

def sample_surface_eta_omega(param, sample_num):
    """

    :param param: a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4])
           index  0   1   2   3   4   5   6   7   8   9   10  11  12  13
    :param sample_num:
    :return:
    """
    def fexp(x, p):
        return torch.sign(x) * (torch.abs(x) ** p)

    eta = torch.rand(sample_num, device=param.device) * torch.pi - torch.pi / 2
    omega = torch.rand(sample_num, device=param.device) * torch.pi * 2 - torch.pi
    x = param[0] * fexp(torch.cos(eta), param[3]) * fexp(torch.cos(omega), param[4])
    y = param[1] * fexp(torch.cos(eta), param[3]) * fexp(torch.sin(omega), param[4])
    z = param[2] * fexp(torch.sin(eta), param[3])
    points = torch.stack((x, y, z)).T
    return points

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
            # # =================================================================
            # # this part will cause nan in model grad and parameters
            # points1 = sample_surface(param, int(sample_num / 2))
            # points2 = sample_surface_eta_omega(param, sample_num)
            # points = torch.cat([points1, points2], dim=0)
            # index = np.arange(points.shape[0])
            # index = np.random.choice(index, sample_num, replace=points.shape[0]<sample_num)
            # points = points[index]
            # # =================================================================
            points = sample_surface_eta_omega(param, sample_num)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
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

def distance_point2superquadric(points, sq_points, params):
    """

    :param points: B N 3
    :param sq_points: B M L 3, L is num of points sampled from superquadrics
    :param params: B M 14:    a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4
    :return:
    """
    # rotate metrixes
    metrix = quaternion_to_matrix(params[..., -4:])  # B M 3 3
    trans = params[..., 7:10]  # B M 3

    B, N, _ = points.shape
    _, M, _, _ = sq_points.shape

    # ==========================================================
    # too many memory occupy!
    # rotate and translate
    points = points.unsqueeze(2)  # B N 1 3
    trans = trans.unsqueeze(1)    # B 1 M 3
    points = torch.einsum('bnoc,bmcd->bnmd', points, metrix) + trans  # B N M 3

    # distance
    points = points.unsqueeze(3)        # B N M 1 3
    sq_points = sq_points.unsqueeze(1)  # B 1 M L 3
    distance = points - sq_points       # B N M L 3
    distance = torch.norm(distance, dim=-1)  # B N M L
    distance = distance.min(dim=-1)[0]  # B N M
    # ==========================================================

    # # ==========================================================
    # # use for loop to reduce occupy
    # distance = torch.zeros(B, N, M, device=points.device)
    # for b in range(B):
    #     for m in range(M):
    #         point = points[b].unsqueeze(1)           # N 1 3
    #         sq_point = sq_points[b, m].unsqueeze(0)  # 1 L 3
    #         rotate_trans_point = torch.matmul(point, metrix[b, m])  # N 1 3
    #         dist = torch.norm(rotate_trans_point - sq_point, dim=-1)  # N L
    #         dist = dist.min(dim=-1)[0]  # N
    #         distance[b, :, m] = dist
    #         del point, sq_point, rotate_trans_point, dist
    #         torch.cuda.empty_cache()
    #         print("for", b, m)
    # # ==========================================================

    return distance

def distance_superquadric2point(points, sq_points, params, one_hot):
    """

    :param points: B N 3
    :param sq_points: B M L 3, L is num of points sampled from superquadrics
    :param params: B M 14:    a1, a2, a3, e1, e2, K1, K2, t1, t2, t3, q1, q2, q3, q4
    :param one_hot: B M N
    :return:
    """
    # rotate metrixes
    metrix = quaternion_to_matrix(params[..., -4:])  # B M 3 3
    trans = params[..., 7:10]  # B M 3

    B, N, _ = points.shape
    _, M, L, _ = sq_points.shape

    # rotate and translate
    points = points.unsqueeze(2)  # B N 1 3
    trans = trans.unsqueeze(1)    # B 1 M 3
    points = torch.einsum('bnoc,bmcd->bnmd', points, metrix) + trans  # B N M 3
    points = points.transpose(2, 1)  # B M N 3

    # distance
    sq_points = sq_points.unsqueeze(3)  # B M L 1 3
    points = points.unsqueeze(2)        # B M 1 N 3
    distance = points - sq_points       # B M L N 3
    distance = torch.norm(distance, dim=-1)  # B M L N
    distance = distance.transpose(2, 1)  # B L M N

    # one hot mask
    one_hot = one_hot.unsqueeze(1).expand(-1, L, -1, -1)  # B L M N
    distance[~one_hot.bool()] = float('inf')  # B L M N
    distance = distance.min(dim=-1)[0]  # B L M

    return distance


def inf_nan_to_num(tensor, num=0.0):
    is_inf = torch.isfinite(tensor)
    is_nan = torch.isfinite(tensor)
    tensor[~is_inf] = num
    tensor[~is_nan] = num
    return tensor

if __name__ == "__main__":
    # sp_param = torch.rand(2, 2, 14).cuda()
    # sp_param.requires_grad = True
    # sample_points = batch_sample(sp_param, 100)
    # print(torch.autograd.grad(torch.sum(sample_points), sp_param))

    points = torch.rand(1, 2, 3).cuda()
    # print(points)
    sq_points = torch.rand(1, 4, 5, 3).cuda()
    sp_param = torch.rand(1, 4, 14).cuda()
    # sp_param = torch.FloatTensor([[[0,0,0,0,0,0,0,1,0,0,1,0,0,1],[0,0,0,0,0,0,0,0,1,0,-1,0,0,1]]]).cuda()
    sp_param.requires_grad = True
    points = distance_p2d(points, sq_points, sp_param)
    # print(points)
    # print(points.shape)
    # print(torch.autograd.grad(torch.sum(points), sp_param))
    pass
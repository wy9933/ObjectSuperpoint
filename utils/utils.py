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
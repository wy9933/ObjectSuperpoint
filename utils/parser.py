import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()

    # args
    parser.add_argument('--gpu', type=int, default=0, help='the number of gpu to use')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--config', type=str, help='yaml config file path')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')

    # train args
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')

    # test args
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--ckpt_path', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--vis', action='store_true', default=False, help='test mode visualize superpoint result')

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpt_path is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    args.experiment_path = os.path.join('./experiments', args.exp_name)
    if args.test:
        args.experiment_path = os.path.join(args.experiment_path, 'test')
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
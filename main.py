import os
import time
import torch
from tensorboardX import SummaryWriter

from utils import parser

from tools.train import train
from tools.test import test
from utils.config import *
from utils.utils import *


def main():
    # args
    args = parser.get_args()

    # CUDA
    torch.cuda.set_device(args.gpu)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    args.log_file = log_file

    # config
    config = get_config(args)

    # log
    log_args_to_file(args, 'args')
    log_config_to_file(args, config, 'config')

    # run
    if args.test:
        test(args, config)
    else:
        summarywriter = SummaryWriter(os.path.join(args.experiment_path, 'TensorboardLog'))
        train(args, config, summarywriter)


if __name__ == '__main__':
    main()

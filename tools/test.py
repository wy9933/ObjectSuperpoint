import os

import torch
import torch.utils.data as data

import numpy as np
import open3d as o3d

from models.superpoint import SuperPoint

from utils.config import *
from utils.utils import *


def test(args, config):
    # build dataset
    if config.dataset.NAME == 'StyleDataset':
        from datasets.StyleDataset import StyleDataset
        test_dataset = StyleDataset(args, config, 'whole')
    else:
        raise NotImplementedError(f'{config.dataset.NAME} not implemented')

    test_loader = data.DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # load model
    model = SuperPoint(config)
    model = model.cuda()
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    is_best = checkpoint['is_best']

    print_log(args, f'Load checkpoint from {args.ckpt_path}, epoch: {epoch}, best_loss: {best_loss}, is_best: {is_best}')

    # Criterion
    criterion = model.get_loss

    # test
    if args.vis:
        vis_dir = os.path.join(args.experiment_path, 'visualize')
        os.makedirs(vis_dir, exist_ok=True)
        np.random.seed(123)
        sp_colors = np.random.rand(config.model.superpoint_num, 3)
    losses = AverageMeter(['loss_fit', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'all_loss'])
    model.eval()
    for i, (class_name, model_id, points) in enumerate(test_loader):
        # run model
        batch_size = points.shape[0]
        points = points.cuda()
        with torch.no_grad():
            p_feat, sp_atten, sp_feat, sp_param = model(points)

        # loss
        loss_fit, loss_ss, loss_loc, loss_sp_balance = criterion(points, p_feat, sp_atten, sp_feat, sp_param)
        loss = 1.0 * loss_fit + 1.0 * loss_ss + 1.0 * loss_loc + 0.001 * loss_sp_balance
        loss /= batch_size

        # summary
        losses.update([loss_fit.item(), loss_ss.item(), loss_loc.item(), loss_sp_balance.item(), loss.item()])
        torch.cuda.empty_cache()

        # visulize
        if args.vis:
            pcd = o3d.geometry.PointCloud()
            for b in range(batch_size):
                vis_file = class_name[b] + '_' + model_id[b] + ".ply"
                vis_path = os.path.join(vis_dir, vis_file)
                points = points[b].cpu().numpy()
                sp_atten = sp_atten[b].cpu().numpy().T
                sp_idx = np.argmax(sp_atten, axis=-1)
                print(vis_file, np.unique(sp_idx, return_counts=True))
                colors = sp_colors[sp_idx].reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(vis_path, pcd)

    print_log(args, '[Test] Losses = %s' % (['%.8f' % l for l in losses.avg()]))
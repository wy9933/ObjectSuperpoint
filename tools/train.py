import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from models.superpoint import SuperPoint

from utils.config import *
from utils.utils import *


def train(args, config, summarywriter):
    # build dataset
    if config.dataset.NAME == 'StyleDataset':
        from datasets.StyleDataset import StyleDataset
        train_dataset = StyleDataset(args, config, 'train')
        val_dataset = StyleDataset(args, config, 'val')
    else:
        raise NotImplementedError(f'{config.dataset.NAME} not implemented')

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.train.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   drop_last=True)

    val_loader = data.DataLoader(train_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # build model
    model = SuperPoint(config)
    model = model.cuda()


    # optimizer & scheduler
    if config.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=config.optimizer.kwargs.lr,
                                weight_decay=config.optimizer.kwargs.weight_decay)
    elif config.optimizer.type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.kwargs.lr)
    else:
        raise NotImplementedError()

    if config.scheduler.type == 'CosLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_epoch)
    else:
        raise NotImplementedError()

    # Criterion
    criterion = model.get_loss

    # TODO: resume training
    pass

    best_loss = float('inf')
    for epoch in range(config.train.max_epoch):
        # train
        losses = train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, summarywriter)
        scheduler.step()

        is_best = False

        summarywriter.add_scalar('Loss/Epoch/loss_fit', losses.avg(0), epoch)
        summarywriter.add_scalar('Loss/Epoch/loss_ss', losses.avg(1), epoch)
        summarywriter.add_scalar('Loss/Epoch/loss_loc', losses.avg(2), epoch)
        summarywriter.add_scalar('Loss/Epoch/loss_sp_balance', losses.avg(3), epoch)
        summarywriter.add_scalar('Loss/Epoch/all_loss', losses.avg(4), epoch)

        # validate
        if (epoch+1) % args.val_freq == 0:
            losses_val = validate(args, model, val_loader, criterion, epoch, summarywriter)

            summarywriter.add_scalar('ValLoss/Epoch/loss_fit', losses_val.avg(0), epoch)
            summarywriter.add_scalar('ValLoss/Epoch/loss_ss', losses_val.avg(1), epoch)
            summarywriter.add_scalar('ValLoss/Epoch/loss_loc', losses_val.avg(2), epoch)
            summarywriter.add_scalar('ValLoss/Epoch/loss_sp_balance', losses_val.avg(3), epoch)
            summarywriter.add_scalar('ValLoss/Epoch/all_loss', losses_val.avg(4), epoch)

            # save best ckpt
            if losses_val.avg(4) < best_loss:
                is_best = True
                best_loss = losses_val.avg(4)
                filename = os.path.join(args.experiment_path, 'model_best.pth')
                print_log(args, f'Saving checkpoint to: {filename}')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'is_best': is_best
                }, filename)

        # save last ckpt
        filename = os.path.join(args.experiment_path, 'model_last.pth')
        print_log(args, f'Saving checkpoint to: {filename}')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'is_best': is_best
        }, filename)

        # save ckpt every ckpt_save_freq epochs
        if (epoch + 1) % config.train.ckpt_save_freq == 0:
            filename = os.path.join(args.experiment_path, f'model_{epoch}.pth')
            print_log(args, f'Saving checkpoint to: {filename}')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'is_best': is_best
            }, filename)

def train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, summarywriter):
    losses = AverageMeter(['loss_fit', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'all_loss'])
    n_batches = len(train_loader)

    model.train()
    for i, (class_name, model_id, points) in enumerate(train_loader):
        # for name, param in model.named_parameters():
        #     # print(name, param, param.grad)
        #     if param.requires_grad and torch.isnan(param).any():
        #         print(f"梯度NaN detected for parameter: {name}")
        # run model
        batch_size = points.shape[0]
        points = points.cuda()
        p_feat, sp_atten, sp_feat, sp_param = model(points)

        # loss and backward
        loss_fit, loss_ss, loss_loc, loss_sp_balance = criterion(points, p_feat, sp_atten, sp_feat, sp_param)
        loss = 1.0 * loss_fit + 1.0 * loss_ss + 1.0 * loss_loc + 0.001 * loss_sp_balance
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # =========================================================================
        # # for test
        # print("11111")
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(name)
        # nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type=2)
        # optimizer.step()
        # print("22222")
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(name)
        # print("33333")
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(name)
        # # =========================================================================

        # summary
        losses.update([loss_fit.item(), loss_ss.item(), loss_loc.item(), loss_sp_balance.item(), loss.item()])
        n_itr = epoch * n_batches + i
        summarywriter.add_scalar('Loss/Batch/loss_fit', loss_fit.item(), n_itr)
        summarywriter.add_scalar('Loss/Batch/loss_ss', loss_ss.item(), n_itr)
        summarywriter.add_scalar('Loss/Batch/loss_loc', loss_loc.item(), n_itr)
        summarywriter.add_scalar('Loss/Batch/loss_sp_balance', loss_sp_balance.item(), n_itr)
        summarywriter.add_scalar('Loss/Batch/all_loss', loss.item(), n_itr)
        summarywriter.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        torch.cuda.empty_cache()

        # message output
        if (i+1) % 20 == 0:
            print_log(args,
                      '[Epoch %d/%d][Batch %d/%d] Losses = %s lr = %.6f' %
                      (epoch, config.train.max_epoch, i + 1, n_batches,
                       ['%.8f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))

    print_log(args,
              '[Training] EPOCH: %d Losses = %s' %
              (epoch, ['%.8f' % l for l in losses.avg()]))

    return losses


def validate(args, model, val_loader, criterion, epoch, summarywriter):
    print_log(args, f"Start validating epoch {epoch}")
    losses = AverageMeter(['loss_fit', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'all_loss'])
    n_batches = len(val_loader)

    model.eval()
    for i, (class_name, model_id, points) in enumerate(val_loader):
        # run model
        batch_size = points.shape[0]
        points = points.cuda()
        p_feat, sp_atten, sp_feat, sp_param = model(points)

        # loss
        loss_fit, loss_ss, loss_loc, loss_sp_balance = criterion(points, p_feat, sp_atten, sp_feat, sp_param)
        loss = 1.0 * loss_fit + 1.0 * loss_ss + 1.0 * loss_loc + 0.001 * loss_sp_balance
        loss /= batch_size

        # summary
        losses.update([loss_fit.item(), loss_ss.item(), loss_loc.item(), loss_sp_balance.item(), loss.item()])

        torch.cuda.empty_cache()

        # message output
        if (i + 1) % 50 == 0:
            print_log(args,
                      'Validate [%d/%d] Losses = %s' %
                      (i + 1, n_batches, ['%.8f' % l for l in losses.val()]))

        # visual output
        if i == 0 and(epoch + 1) % 25 == 0:
            colors = torch.rand(points.shape) * 255
            summarywriter.add_mesh('val/pointcloud', points, colors, global_step=epoch)

    print_log(args,
              '[Validate] EPOCH: %d Losses = %s' %
              (epoch, ['%.8f' % l for l in losses.avg()]))

    return losses
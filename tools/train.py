import torch
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
        train_dataset = StyleDataset(config, 'train')
        val_dataset = StyleDataset(config, 'val')
    else:
        raise NotImplementedError(f'{config.dataset.NAME} not implemented')

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.train.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   drop_last=True)

    val_loader = data.DataLoader(train_dataset,
                                 batch_size=config.train.batch_size,
                                 shuffle=True,
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
    else:
        raise NotImplementedError()

    if config.scheduler.type == 'CosLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler.kwargs.epochs)
    else:
        raise NotImplementedError()

    # Criterion
    # TODO
    pass

    # TODO: resume training
    pass


def train_one_epoch(model, train_loder, optimizer, criterion, epoch):
    model.train()
    for i, (class_name, model_id, points) in enumerate(train_loder):
        points = points.cuda()
        p_feat, sp_atten, sp_feat, sp_param = model(points)

        loss = criterion(points, p_feat, sp_atten, sp_feat, sp_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

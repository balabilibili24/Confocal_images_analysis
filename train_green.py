# coding:utf8
from config import opt
import os
import random
import models


from dataset.dataset_green import dataset_green
from utils.dice_loss import DiceLoss
from utils.visualize import Visualizer

from torchnet import meter
from torch.optim import lr_scheduler

import torch as t
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from torchvision import transforms

# loss函数：分割
seg_criterion = DiceLoss()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    if not os.path.exists('/home/lz/sgd/checkpoints/{}/{}--{}--{}/'.format(opt.model, opt.mark, opt.lr, opt.batch_size)):
        os.makedirs('/home/lz/sgd/checkpoints/{}/{}--{}--{}/'.format(opt.model, opt.mark, opt.lr, opt.batch_size))

    # configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # ************多GPU*****************
    # model = t.nn.DataParallel(model, device_ids=opt.device_ids)

    # data
    train_data = dataset_green(opt.train_data_root, train=True)
    val_data = dataset_green(opt.train_data_root, val=True)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr, betas=(0.9, 0.99), weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2)

    iter_num = 0
    max_iterations = opt.max_epoch * len(train_dataloader)

    # meters
    loss_meter = meter.AverageValueMeter()
    dice_train = meter.AverageValueMeter()

    train_loss = []
    train_DICE = []
    val_DICE = []

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        dice_train.reset()

        for ii, (data, mask) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            mask = mask.to(opt.device)

            seg_out = model(input)
            loss = seg_criterion(seg_out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # meters update and visualize
            loss_meter.add(loss.item())

            seg_score = (seg_out > 0.5).float()
            seg_dice = seg_criterion(seg_score, mask)
            dice_train.add(seg_dice.item())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        # lr = optimizer.param_groups[0]['lr']

        # test
        train_dice = 1 - dice_train.value()[0]
        val_dice = val(model, val_dataloader)

        train_loss.append(loss_meter.value()[0])
        train_DICE.append(train_dice)
        val_DICE.append(val_dice)

        np.save('/home/lz/sgd/checkpoints/{}/{}--{}--{}/train_loss.npy'.format(opt.model, opt.mark, opt.lr,
                                                                                        opt.batch_size), train_loss)
        np.save('/home/lz/sgd/checkpoints/{}/{}--{}--{}/train_dice.npy'.format(opt.model, opt.mark, opt.lr,
                                                                                        opt.batch_size), train_DICE)
        np.save('/home/lz/sgd/checkpoints/{}/{}--{}--{}/val_dice.npy'.format(opt.model, opt.mark, opt.lr,
                                                                                      opt.batch_size), val_DICE)

        model.save(
            '/home/lz/sgd/checkpoints/{}/{}--{}--{}/{:0>4d}--{:.4f}--{:.4f}.pth'.format(
                opt.model, opt.mark, opt.lr, opt.batch_size, epoch, train_dice, val_dice))

        vis.plot('train_dice', train_dice)
        vis.plot('val_dice', val_dice)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_dice:{train_dice},val_dice:{val_dice}".format(
            epoch=epoch, lr=lr, loss=loss_meter.value()[0], train_dice=train_dice, val_dice=val_dice))

        # # update learning rate
        # previous_loss = loss_meter.value()[0]
        # scheduler.step(previous_loss)


@t.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    val_dice = meter.AverageValueMeter()
    for ii, (val_input, mask) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        mask = mask.to(opt.device)

        seg_out = model(val_input)
        seg_score = (seg_out > 0.5).float()
        seg_dice = seg_criterion(seg_score, mask)
        val_dice.add(seg_dice.item())

    model.train()

    return 1 - val_dice.value()[0]


def mkdir(path):
    '''make dir'''

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
        print("****加载成功****")
    model.to(opt.device)

    # data
    test_data = dataset_green(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    test_dice = meter.AverageValueMeter()
    for ii, (test_input, mask) in tqdm(enumerate(test_dataloader)):
        test_input = test_input.to(opt.device)
        mask = mask.to(opt.device)

        seg_out = model(test_input)
        seg_score = (seg_out > 0.5).float()

        seg_mask = seg_score.squeeze().cpu().numpy() * 255

        import cv2
        cv2.imwrite('/home/lz/sgd/{}.png'.format(ii), seg_mask)

        # print(np.max(seg_mask), np.min(seg_mask))

    #     seg_dice = seg_criterion(seg_score, mask)
    #     test_dice.add(seg_dice.item())
    #
    # dice = 1 - test_dice.value()[0]
    #
    # print("test_dice: {}".format(dice))


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # 设置随机数种子
    set_seed(20)
    import fire

    fire.Fire()

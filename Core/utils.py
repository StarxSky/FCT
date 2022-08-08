# 功能实现文件
# 包括： 图像处理,
#

import os
import torch
import pickle
import numpy as np


from tqdm import tqdm
from PIL import Image
from Core.Config import ResNet_50_new_Config
from torchvision.datasets import CIFAR100
from typing import Callable, Optional


# 保存图像数据
def save_pngs(untar_path: str, spilt: str):
    spilt_map = {'train': 'training', 'test': 'validation'}
    spilt_path = os.path.join(untar_path, spilt_map.get(spilt))
    os.makedirs(spilt_path, exist_ok=True)

    for i in range(100):
        class_path = os.path.join(spilt_path, str(i))
        os.makedirs(class_path, exist_ok=True)

    with open(os.path.join(untar_path, spilt), 'rb') as f:
        data_path = pickle.load(f, encoding='latin1')
    data = data_path.get('data')  # numpy array
    # Reshape and cast
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.transpose(0, 2, 3, 1).astype('uint8')

    labels = data_path.get('fine_labels')
    for i, (datum, label) in tqdm(enumerate(zip(data, labels)), total=len(labels)):
        images = Image.fromarray(datum)
        images = images.convert('RGB')
        file_path = os.path.join(spilt_path, str(label), '{}.png'.format(i))
        images.save(file_path)


#
# 获取数据
def get_cifar100() -> None:
    """Get and reformat cifar100 dataset.

    See https://www.cs.toronto.edu/~kriz/cifar.html for dataset description.
    """
    data_store_dir = 'data_store'

    if not os.path.exists(data_store_dir):
        os.makedirs(data_store_dir)

    dataset = CIFAR100(root=data_store_dir, download=True)

    # Load files and convert to PNG
    untar_dir = os.path.join(data_store_dir, dataset.base_folder)
    # 分别将不同图像进行分类
    save_pngs(untar_dir, 'test')
    save_pngs(untar_dir, 'train')







#
# 获取优化器
def get_optimizer(model,
                  algorithm: str,
                  lr: float,
                  weight_decay: float,
                  momentum: float = None,
                  no_bn_decay: bool = False,
                  nesterov: bool = False):

    # 使用SGD优化器
    if algorithm == 'sgd':
        parameters = list(model.named_parameters())
        # BatchNormal层参数
        bn_params = [v for n, v in parameters if ('bn' in n) and v.requires_grad]
        # 重置参数
        rest_params = [v for n, v in parameters if ('bn' not in n) and v.requires_grad]

        # 设定优化器
        optimizer = torch.optim.SGD([
            {'params': bn_params, 'weight_decay': 0 if no_bn_decay else weight_decay},
            {'params': rest_params, 'weight_decay': weight_decay}],
            lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    # 使用Adam优化器
    elif algorithm == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)

    return optimizer


#
#
def get_policy(optimizer):
    """Get learning policy given its configurations.

    :param optimizer: A torch optimizer.
    :param algorithm: Name of the learning rate scheduling algorithm.
    :return: A callable to adjust learning rate for each epoch.
    """

    return cosine_lr(optimizer)


#
#
def assign_learning_rate(optimizer: torch.optim.Optimizer,
                         new_lr: float) -> None:
    """Update lr parameter of an optimizer.

    :param optimizer: A torch optimizer.
    :param new_lr: updated value of learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


#
#
def _warmup_lr(base_lr: float,
               warmup_length: int,
               epoch: int) -> float:
    """Get updated lr after applying initial warmup.

    :param base_lr: Nominal learning rate.
    :param warmup_length: Number of epochs for initial warmup.
    :param epoch: Epoch number.
    :return: Warmup-updated learning rate.
    """
    return base_lr * (epoch + 1) / warmup_length



#
#, ,

def cosine_lr(optimizer,
              warmup_length=5,
              epochs=100,
              lr=1.024,
              **kwargs) -> Callable:
    """Get lr adjustment callable with cosine schedule.

    :param optimizer: A torch optimizer.
    :param warmup_length: Number of epochs for initial warmup.
    :param epochs: Epoch number.
    :param lr: Nominal learning rate value.
    :return: A callable to adjust learning rate per epoch.
    """

    def _lr_adjuster(epoch) -> float:
        """Get updated learning rate.

        :param epoch: Epoch number.
        :param iteration: Iteration number.
        :return: Updated learning rate value.
        """
        if epoch < warmup_length:
            new_lr = _warmup_lr(lr, warmup_length, epoch)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            new_lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr

        assign_learning_rate(optimizer, new_lr)

        return new_lr

    return _lr_adjuster


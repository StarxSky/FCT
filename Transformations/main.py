import os
import torch

from torch import nn
from model import MLP_BN_SIDE_PROJECTION
from Core.Config import TS_Config
from Core.utils import get_policy
from Core.utils import get_optimizer
from Core.Dataset import SubImageFolder
from trainer import TransformationTrainer
from Core.Layers import transformation_to_torchscripts


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

model = MLP_BN_SIDE_PROJECTION(TS_Config.old_embedding_dim,
                               TS_Config.new_embedding_dim,
                               TS_Config.side_info_dim,
                               TS_Config.inner_dim)

old_model = torch.jit.load(TS_Config.old_model_path)
new_model = torch.jit.load(TS_Config.new_model_path)


optimizer = get_optimizer(model,
                               TS_Config.optimizer_algorithm,
                               TS_Config.optimizer_lr,
                               TS_Config.weight_decay)

data = SubImageFolder('cifar100',
                          batch_size=TS_Config.batch_size,
                          data_root='data_store\\cifar-100-python',
                          num_workers=TS_Config.num_workers,
                          num_classes=TS_Config.dataset_num_classes)

lr_policy = get_policy(optimizer)

criterion = nn.MSELoss()


def main(model, new_model, old_model, data, criterion, optimizer, lr_policy) -> None:
    """Run training.

    :param config: A dictionary with all configurations to run training.
    :return:
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        old_model = torch.nn.DataParallel(old_model)
        new_model = torch.nn.DataParallel(new_model)

    model.to(device)
    old_model.to(device)
    new_model.to(device)

    if TS_Config.side_info_model_path is not None:
        side_info_model = torch.jit.load(TS_Config.side_info_model_path)
        if torch.cuda.is_available():
            side_info_model = torch.nn.DataParallel(side_info_model)
        side_info_model.to(device)
    else:
        side_info_model = old_model

    trainer = TransformationTrainer(old_model, new_model, side_info_model)
    for epoch in range(TS_Config.epochs):
        lr_policy(epoch, iteration=None)

        if TS_Config.switch_mode_to_eval:
            switch_mode_to_eval = epoch >= TS_Config.epochs / 2
        else:
            switch_mode_to_eval = False

        train_loss = trainer.train(
            train_loader=data.train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            switch_mode_to_eval=switch_mode_to_eval,
        )

        print("Train: epoch = {}, Average Loss = {}".format(epoch, train_loss))

        # evaluate on validation set
        test_loss = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        print("Test: epoch = {}, Average Loss = {}".format(
            epoch, test_loss
        ))

    transformation_to_torchscripts(old_model, side_info_model, model,
                                   TS_Config.output_transformation_path,
                                   TS_Config.output_transformed_old_model_path)



if __name__ == "__main__" :
    main(model, new_model, old_model, data, criterion, optimizer, lr_policy)


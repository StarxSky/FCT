import torch

from model import ResNet50
from Core.utils import get_policy
from trainer import BackboneTrainer
from Core.utils import get_optimizer
from Core.Layers import LabelSmoothing
from Core.Dataset import SubImageFolder
from Core.Config import ResNet_50_old_Config
from Core.Layers import backbone_to_torchscript


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

model = ResNet50(ResNet_50_old_Config.model_num_classes,
                 ResNet_50_old_Config.embedding_dim,
                 ResNet_50_old_Config.last_nonlin)


optimizer = get_optimizer(model,
                          ResNet_50_old_Config.optimizer_algorithm,
                          ResNet_50_old_Config.optimizer_lr,
                          ResNet_50_old_Config.weight_decay,
                          ResNet_50_old_Config.momentum,
                          ResNet_50_old_Config.no_bn_decay,
                          ResNet_50_old_Config.nesterov)

lr_policy = get_policy(optimizer)

criterion = LabelSmoothing(ResNet_50_old_Config.label_smoothing)


data = SubImageFolder('cifar100',
                      batch_size = ResNet_50_old_Config.batch_size,
                      data_root = 'data_store\\cifar-100-python',
                      num_workers=ResNet_50_old_Config.num_workers,
                      num_classes=ResNet_50_old_Config.dataset_num_classes)


def main(model, model_path, data, criterion, optimizer, lr_policy, epochs) :
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    trainer = BackboneTrainer()
    # Training loop
    for epoch in range(epochs):
        lr_policy(epoch)

        train_acc1, train_acc5, train_loss = trainer.train(
            train_loader=data.train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(
            "Train: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, train_loss, train_acc1, train_acc5
            ))

        test_acc1, test_acc5, test_loss = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        print(
            "Test: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, test_loss, test_acc1, test_acc5
            ))

    backbone_to_torchscript(model, model_path)


if __name__ == "__main__" :
    main(model,
         data = data,
         model_path=ResNet_50_old_Config.output_model_path,
         criterion=criterion,
         optimizer=optimizer,
         lr_policy=lr_policy,
         epochs=ResNet_50_old_Config.epochs)



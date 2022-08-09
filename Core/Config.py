import os 

datas_path = str(input('dataset file name (eg. : data_store/cifar-100-python):'))# eg. : data_store/cifar-100-python
os.makedirs('checkpoints', exist_ok=True)
checkpoints = 'checkpoints'


class ResNet_50_old_Config():
    model_num_classes = 100
    dataset_num_classes = 50
    num_workers = 20
    batch_size = 1024
    embedding_dim = 128
    last_nonlin = True
    optimizer_algorithm = 'sgd'
    optimizer_lr = 1.024
    lr_policy = 1.024
    warmup_length = 5
    epochs_policy = 100
    epochs = 100
    weight_decay = 0.000030517578125
    no_bn_decay = False
    momentum = 0.875
    nesterov = False
    label_smoothing= 0.1
    output_model_path= f"{checkpoints}/cifar100_old.pt"


class ResNet_50_new_Config():
    model_num_classes = 100
    dataset_num_classes = 100
    num_workers = 20
    batch_size = 1024
    embedding_dim = 128
    last_nonlin = True
    optimizer_algorithm = 'sgd'
    optimizer_lr = 1.024
    lr_policy = 1.024
    warmup_length = 5
    epochs_policy = 100
    epochs = 100
    weight_decay = 0.000030517578125
    no_bn_decay = False
    momentum = 0.875
    nesterov = False
    label_smoothing= 0.1
    output_model_path= f"{checkpoints}/cifar100_new.pt"


class TS_Config():
    old_embedding_dim = 128
    new_embedding_dim =128
    side_info_dim = 128
    inner_dim = 2048
    optimizer_algorithm = 'adam'
    optimizer_lr= 0.0005
    weight_decay= 0.000030517578125
    old_model_path= f'{checkpoints}/cifar100_old.pt'
    new_model_path= f'{checkpoints}/cifar100_new.pt'
    side_info_model_path= f'{checkpoints}/imagenet_1000_simclr.pt' 

    name= 'cifar100'  
    dataset_num_classes= 100  # This is the number of classes to include for training.
    num_workers= 20
    batch_size= 1024

    policy_algorithm= 'cosine_lr'
    warmup_length= 5
    policy_epochs= 80
    policy_lr= 0.0005

    epochs= 80
    switch_mode_to_eval= True
    output_transformation_path= f'{checkpoints}/cifar100_transformation.pt'
    output_transformed_old_model_path= f'{checkpoints}/cifar100_old_transformed.pt'

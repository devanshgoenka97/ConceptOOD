import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from util.dataset_largescale import DatasetWithMeta
from util.broden_loader import BrodenDataset, broden_collate, dataloader

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'imagenet': 1000}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule = None, None, [50, 75, 90]
    if args.in_dataset == "imagenet":
        root = '/home/sunyiyou/dataset/imagenet'
        # Data loading code
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)

    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes_dict[args.in_dataset],
    })

def get_loader_out(args, dataset=('dtd'), config_type='default', split=('val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    val_ood_loader = None

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
        if val_dataset == 'dtd':
            transform = config.transform_test_largescale if args.in_dataset in {'imagenet'} else config.transform_test
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images", transform=transform),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'places365':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/", transform=transform_test),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'inat':
            val_ood_loader = torch.utils.data.DataLoader(
                DatasetWithMeta('/media/sunyiyou/ubuntu-hdd1/dataset/iNat',
                                '/media/sunyiyou/ubuntu-hdd1/dataset/iNat/inat_plantae_selected_list_nolabel.txt', config.transform_test_largescale),
                batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'imagenet':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join('/home/sunyiyou/dataset/imagenet', 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        else:
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(val_dataset),
                                                          transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=2)

    return EasyDict({
        "val_ood_loader": val_ood_loader,
    })

def get_loader_probe(args):
    transform = {
        'imagenet': transform_test_largescale,
    }[args.in_dataset]
    dataset = BrodenDataset('/nobackup/broden1_224',
                            categories=["object", "part", "scene", "material", "texture", "color"], transform=transform)
    return dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=broden_collate)


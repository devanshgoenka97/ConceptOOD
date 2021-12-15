"""
Script to visualize some sample images from the datasets
"""
from __future__ import print_function

import torch
import torchvision
from torchvision import transforms
import numpy as np
from util.args_loader import get_args
import matplotlib.pyplot as plt

import numpy as np
import pickle
import os

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

args = get_args()


# ImageNet transformation
transform_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inv_transform = transforms.Normalize(
mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
 )

toPIL = transforms.ToPILImage()

# Using de-depulicated iNaturalist, SUN, Places and Textures as the OOD datasets
inat_dataset = torchvision.datasets.ImageFolder('/nobackup/iNaturalist', transform_largescale)
sun_dataset = torchvision.datasets.ImageFolder('/nobackup/SUN', transform_largescale)
places_dataset = torchvision.datasets.ImageFolder('/nobackup/Places', transform_largescale)
dtd_dataset = torchvision.datasets.ImageFolder('/nobackup/dtd/images', transform_largescale)

ood_datasets = [('inat', inat_dataset), ('sun', sun_dataset), ('places', places_dataset), ('dtd', dtd_dataset)]


with torch.no_grad():
    for name, dataset in ood_datasets:
        ood_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        for batch, data in enumerate(ood_loader):
            ood_input, _ = data
            if name =='inat':
                image = toPIL(inv_transform(ood_input[12]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 12))
                image = toPIL(inv_transform(ood_input[13]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 13))
            elif name == 'sun':
                image = toPIL(inv_transform(ood_input[12]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 12))
                image = toPIL(inv_transform(ood_input[13]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 13))
            elif name == 'places':
                image = toPIL(inv_transform(ood_input[11]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 11))
                image = toPIL(inv_transform(ood_input[13]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 13))
            else:
                image = toPIL(inv_transform(ood_input[34]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 34))
                image = toPIL(inv_transform(ood_input[78]))
                image.save('broden/results/ood_{0}_{1}.png'.format(name, 78))
            break
            

print("Done generating samples")

                
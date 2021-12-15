from __future__ import print_function
from typing import OrderedDict

from models.resnet import resnet18

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from util.args_loader import get_args

import numpy as np
import pickle
import os
import csv

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

# Using iNaturalist as the OOD dataset
inat_dataset = torchvision.datasets.ImageFolder('/nobackup/iNaturalist', transform_largescale)
sun_dataset = torchvision.datasets.ImageFolder('/nobackup/SUN', transform_largescale)
places_dataset = torchvision.datasets.ImageFolder('/nobackup/Places', transform_largescale)

# ResNet based on pre-trained ImageNet
model = resnet18(pretrained=True)
model.cuda()
model.eval()

# Only reading the object file to extract semantics
concepts = []
with open('/nobackup/broden1_224/c_object.csv', 'r') as f:
    csvreader = csv.reader(f)
    # Elegantly ignoring first row in CSV file
    skip  = 0
    while(skip >= 0):
        next(csvreader)
        skip -= 1
    for row in csvreader:
        concepts.append(int(row[1]))


inat_loader = torch.utils.data.DataLoader(inat_dataset, batch_size=100, shuffle=False)
places_loader = torch.utils.data.DataLoader(places_dataset, batch_size=100, shuffle=False)
sun_loader = torch.utils.data.DataLoader(sun_dataset, batch_size=100, shuffle=False)

ood_features = {'inat': [], 'sun': [], 'places': []}

with torch.no_grad():
            
    for j, data in enumerate(inat_loader):
        inputs, _ = data
        inputs = inputs.cuda()

        # Forward propagating inputs through pre-trained network
        outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)

        outputs = outputs.to(torch.float32)
        ood_features['inat'].extend(outputs.detach().cpu().numpy())
        break

    for j, data in enumerate(places_loader):
        inputs, _ = data
        inputs = inputs.cuda()

        # Forward propagating inputs through pre-trained network
        outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)

        outputs = outputs.to(torch.float32)
        ood_features['places'].extend(outputs.detach().cpu().numpy())
        break

    for j, data in enumerate(sun_loader):
        inputs, _ = data
        inputs = inputs.cuda()

        # Forward propagating inputs through pre-trained network
        outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)

        outputs = outputs.to(torch.float32)
        ood_features['sun'].extend(outputs.detach().cpu().numpy())
        break

print("Done retrieving feature vectors for all OOD samples")

ood_features['sun'] = np.array(ood_features['sun'])
ood_features['places'] = np.array(ood_features['places'])
ood_features['inat'] = np.array(ood_features['inat'])

# Saving results to file
save_dir = os.path.join('broden', 'results')
os.makedirs(save_dir, exist_ok=True)

save_file = "ood_features.pkl"
with open(os.path.join(save_dir, save_file), 'wb') as f:
    pickle.dump(ood_features, f)

                
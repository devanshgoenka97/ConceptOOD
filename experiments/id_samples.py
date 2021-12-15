from __future__ import print_function

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

# Using ImageNet 2012 val set as the dataset
dataset = torchvision.datasets.ImageFolder('/nobackup/imagenet_extracted', transform_largescale)

# Taking a subset of ImageNet
targets = [363, 295, 296, 297, 298, 299, 281, 283, 284, 285, 286, 288, 290, 7, 8, 3, 2, 18, 21, 49, 50, 158, \
    160, 161, 162, 182, 207, 208, 219, 235, 236, 246, 250, 270, 272, 322, 323, 330, 331, 333, 337, 341, 346, \
        361, 366, 367, 368, 385, 387,  388]

# ResNet based on pre-trained ImageNet
model = resnet18(pretrained=True)
model.cuda()
model.eval()

result_log = {}

# For each target class, retrieve the class conditional concept probability
for i, target in enumerate(targets):

    # Subset the dataset to contain only that class
    target_indices = [i for i, label in enumerate(dataset.targets) if label == target]
    target_subset = torch.utils.data.Subset(dataset, target_indices)
    val_loader = torch.utils.data.DataLoader(target_subset, batch_size=50, shuffle=False)

    # Creating a dictionary for each concept
    result_log[target] = []

    with torch.no_grad():
            
        for j, data in enumerate(val_loader):
            inputs, _ = data
            inputs = inputs.cuda()

            # Forward propagating inputs through pre-trained network
            outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)
            outputs = outputs.to(torch.float32)
            result_log[target].extend(outputs.detach().cpu().numpy())
            break

    result_log[target] = np.array(result_log[target])

print("Done retrieving in-distribution scores for all targets")

# Saving results to file
save_dir = os.path.join('broden', 'results')
os.makedirs(save_dir, exist_ok=True)

save_file = "id_features.pkl"
with open(os.path.join(save_dir, save_file), 'wb') as f:
    pickle.dump(result_log, f)

                
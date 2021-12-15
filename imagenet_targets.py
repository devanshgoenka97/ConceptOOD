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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
dataset = torchvision.datasets.ImageFolder('/nobackup/ImageNet/train', transform_largescale)

# Taking a subset of ImageNet
targets = [363, 295, 296, 297, 298, 299, 281, 283, 284, 285, 286, 288, 290, 7, 8, 3, 2, 18, 21, 49, 50, 158, \
    160, 161, 162, 182, 207, 208, 219, 235, 236, 246, 250, 270, 272, 322, 323, 330, 331, 333, 337, 341, 346, \
        361, 366, 367, 368, 385, 387,  388]

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

k = 10

result_log = {}

# Load the binary classifier
layer = torch.nn.Linear(512, 1).cuda()

# For each target class, retrieve the class conditional concept probability
for i, target in enumerate(targets):

    # Subset the dataset to contain only 1000 samples from each class
    target_indices = [i for i, label in enumerate(dataset.targets) if label == target][:1000]
    target_subset = torch.utils.data.Subset(dataset, target_indices)
    val_loader = torch.utils.data.DataLoader(target_subset, batch_size=1024, shuffle=True)

    print(f"Probing in-distribution data for label : {target}")

    # Creating a dictionary for each concept
    result_log[target] = {}        

    with torch.no_grad():
            
        for j, data in enumerate(val_loader):
            inputs, _ = data
            inputs = inputs.cuda()

            # Forward propagating inputs through pre-trained network
            outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)

            # Iterate over each classifier to obtain concept probability
            for i, concept in enumerate(concepts):
                if not os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
                    print("Classifier was not trained due to too few samples")
                    continue
                
                print(f"Retrieving in-distribution score for concept : {concept}")

                if concept not in result_log[target]:
                    result_log[target][concept] = []

                checkpoint = torch.load("broden/classifiers_2/concept_classifier_{0}.pth".format(concept))
                layer.load_state_dict(checkpoint['model_state_dict'])
                layer.eval()

                # Then propagating the extraced feature vector through binary classifier
                classifier_outputs = torch.sigmoid(layer(outputs)).squeeze()
                classifier_outputs = classifier_outputs.to(torch.float32)

                # Storing the posterior probability for each image
                posterior_probabilities = classifier_outputs.detach().cpu().numpy().ravel()
                result_log[target][concept].extend(posterior_probabilities)

    print(f"Done retrieving in-distribution score for label : {target}")

print("Done retrieving in-distribution scores for all targets")

# Saving results to file
save_dir = os.path.join('broden', 'results')
os.makedirs(save_dir, exist_ok=True)

save_file = "id_score_train.pkl"
with open(os.path.join(save_dir, save_file), 'wb') as f:
    pickle.dump(result_log, f)

                
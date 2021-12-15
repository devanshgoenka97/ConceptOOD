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

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ImageNet transformation
transform_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Using de-depulicated iNaturalist, SUN, Places and Textures as the OOD datasets
inat_dataset = torchvision.datasets.ImageFolder('/nobackup/iNaturalist', transform_largescale)
sun_dataset = torchvision.datasets.ImageFolder('/nobackup/SUN', transform_largescale)
places_dataset = torchvision.datasets.ImageFolder('/nobackup/Places', transform_largescale)
dtd_dataset = torchvision.datasets.ImageFolder('/nobackup/dtd/images', transform_largescale)

ood_datasets = [('inat', inat_dataset), ('sun', sun_dataset), ('places', places_dataset), ('dtd', dtd_dataset)]

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

ood_scores = {}

with torch.no_grad():
    for name, dataset in ood_datasets:
        ood_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

        print(f"Processing OOD Dataset: {name}")

        ood_scores[name] = {}

        for batch, data in enumerate(ood_loader):

            ood_input, _ = data
            ood_input = ood_input.cuda()

            # Forward propagating inputs through pre-trained network
            outputs = F.avg_pool2d(model.features(ood_input), 7).view(ood_input.shape[0], 512)
            
            for i, concept in enumerate(concepts):

                # Load the trained binary classifier for the given concept
                layer = torch.nn.Linear(512, 1).cuda()
                
                if not os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
                    print("Classifier was not trained due to too few samples")
                    continue
                
                checkpoint = torch.load("broden/classifiers_2/concept_classifier_{0}.pth".format(concept))
                layer.load_state_dict(checkpoint['model_state_dict'])
                layer.eval()

                # If concept does not exist in dictionary from before
                if concept not in ood_scores[name]:
                    ood_scores[name][concept] = []
                
                print(f"Retrieving out-of-distribution score for concept : {concept}")

                # Then propagating the extraced feature vector through binary classifier
                classifier_output = torch.sigmoid(layer(outputs)).squeeze()
                classifier_output = classifier_output.to(torch.float32)

                # Storing the posterior probability for each image
                posterior_probabilities = classifier_output.detach().cpu().numpy().ravel()
                ood_scores[name][concept].extend(np.array(posterior_probabilities))

print("Done retrieving scores for all OOD samples")

# Saving results to file
save_dir = os.path.join('broden', 'results')
os.makedirs(save_dir, exist_ok=True)

save_file = "ood_scores_new.pkl"
with open(os.path.join(save_dir, save_file), 'wb') as f:
    pickle.dump(ood_scores, f)

                
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from util.args_loader import get_args
import os
import numpy as np
import csv
import time
import pickle

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

layer = nn.Linear(512, 1)

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

weights = np.empty(512)

indices = []
counter = 0

for i, concept in enumerate(concepts):

    if not os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
        print("Classifier was not trained due to too few samples")
        continue

    checkpoint = torch.load("broden/classifiers_2/concept_classifier_{0}.pth".format(concept), map_location='cpu')
    layer.load_state_dict(checkpoint['model_state_dict'])

    # Extracting the trained weight vector from the trained classifier
    weight = layer.weight.detach().numpy()

    # Adding to list of weights
    weights = np.vstack((weights, weight))

    # Keeping track of the index at which the concept exists in the weight array
    indices.append(concept)
    counter += 1

# Removing the first empty array
weights = weights[1:]

# Normalize the weights
lengths = (weights**2).sum(axis=1, keepdims=True)**.5
weights = weights/lengths

save_struct = {'weights': weights, 'indices': indices}

with open('broden/results/weights_2.pkl', 'wb') as f:
    pickle.dump(save_struct, f)

concept_dog = weights[indices.index(93)]

similarity = concept_dog.dot(weights.T)

# Closest concepts to dog

import pdb; pdb.set_trace()

from __future__ import print_function

import torch
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

true_concepts = []
# Filtering out concepts that are not trained on
for i, concept in enumerate(concepts):
    if not os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
        continue
    true_concepts.append(concept)

true_concepts = np.array(true_concepts)

# Fetching stored OOD scores
ood_features = {'inat': {}, 'sun': {}, 'places': {}}
with open('broden/results/ood_scores_1000.pkl', 'rb') as f:
    ood_scores = pickle.load(f)

k = 10

closest_concepts = {'inat': [None] * 1000, 'sun': [None] * 1000, 'places': [None] * 1000}

# For each OOD dataset
for ood_set in ood_scores:
    # For each image in OOD dataset
    for index in range(1000):
        probs = []

        # For each concept score of the image
        for concept in ood_scores[ood_set]:
            probs.append(ood_scores[ood_set][concept][index])

        # Getting top k probs for image in OOD set
        topk = torch.tensor(probs).topk(k)
        topconcepts = true_concepts[topk.indices]
        closest_concepts[ood_set][index] = topconcepts


import pdb; pdb.set_trace()
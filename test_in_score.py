from __future__ import print_function

import numpy as np
import pickle
import os
import csv

np.random.seed(1)

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

# Fetching stored ID scores
with open('broden/results/id_score.pkl', 'rb') as f:
    in_scores = pickle.load(f)

k = 10

# For each target in ID dataset
for target in in_scores:
    # For each concept in target
    for concept in in_scores[target]:
        in_scores[target][concept] = in_scores[target][concept].mean()

# For each target in ID dataset
for target in in_scores:
    concept_scores = [in_scores[target][concept] for concept in in_scores[target]]
    # Flattening out structure for easier access
    in_scores[target] = np.array(concept_scores)

# Getting k-closest concepts to each target in the ID dataset
closest_concepts = {}
for target in in_scores:
    closest_concepts[target] = true_concepts[np.argsort(-in_scores[target])[:k]]

import pdb; pdb.set_trace()
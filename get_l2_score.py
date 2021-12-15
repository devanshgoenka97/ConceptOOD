from __future__ import print_function

import numpy as np
import pickle
import os
from util.metrics import cal_metric, plot_distrib

import csv

np.random.seed(1)

# Now we have k-closest concepts to all targets in ID dataset, we can perform retrieve IoU score as follows:
# We will forward propagate our test set through the model to get the k-closest concepts to that sample
# Then we will get the IoU score of the sample against all the k-closest concepts to each target
# Final IoU score will be the maximum among all IoU scores, and then if IoU_max > threshold, it is ID else OOD

def min_euclid_score(sample_probs, in_scores):
    max_min_dist = -1000000000
    # Checking against each target set
    for concept in in_scores:
        max_min_dist = max(max_min_dist, -np.linalg.norm(sample_probs - in_scores[concept]))
    return max_min_dist

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

# Extracting broden labels for human readability
labels = {}
with open('/nobackup/broden1_224/label.csv', 'r') as f:
    csvreader = csv.reader(f)
    # Elegantly ignoring first row in CSV file
    skip  = 0
    while(skip >= 0):
        next(csvreader)
        skip -= 1
    for row in csvreader:
        labels[int(row[0])] = row[1]

true_concepts = []
# Filtering out concepts that are not trained on
for i, concept in enumerate(concepts):
    if not os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
        continue
    true_concepts.append(concept)

true_concepts = np.array(true_concepts)

# Fetching stored training ID scores
with open('broden/results/id_score_train.pkl', 'rb') as f:
    in_scores = pickle.load(f)


# For each target in train ID dataset
for target in in_scores:
    # For each concept in target
    for concept in in_scores[target]:
        # Averaging out the concept posterior probabilities across all samples in ID data
        in_scores[target][concept] = np.array(in_scores[target][concept]).mean()

# For each target in ID dataset
for target in in_scores:
    concept_scores = [in_scores[target][concept] for concept in in_scores[target]]
    # Flattening out structure for easier access
    in_scores[target] = np.array(concept_scores)

# This part propagates each test sample to get the IoU scores for each image in ID dataset.

# Fetching stored validation ID scores
with open('broden/results/id_score_val.pkl', 'rb') as f:
    test_scores = pickle.load(f)

id_dist = []

# For each target in ID dataset
for target in test_scores:
    # For each sample in test target - 50 for each target in ImageNet validation set
    for index in range(50):
        sample_probs = []

        # For each concept in target
        for concept in test_scores[target]:
            sample_probs.append(test_scores[target][concept][index])

        sample_probs = np.array(sample_probs)
        sample_dist = min_euclid_score(sample_probs, in_scores)
        id_dist.append(sample_dist)

id_dist.sort()
threshold = id_dist[round(0.05 * len(id_dist))]

id_ious = np.array(id_dist)

print(f"Threshold is {threshold}")

with open('broden/results/ood_scores_new.pkl', 'rb') as f:
    ood_scores = pickle.load(f)

ood_dists = {}
for ood_set in ood_scores:
    ood_dists[ood_set] = []

# For each OOD dataset
for ood_set in ood_scores:
    # For each image in OOD dataset
    n = len(ood_scores[ood_set][true_concepts[0]])
    for index in range(n):
        probs = []

        # For each concept score of the image
        for concept in ood_scores[ood_set]:
            probs.append(ood_scores[ood_set][concept][index])

        probs = np.array(probs)
        sample_dist = min_euclid_score(probs, in_scores)
        ood_dists[ood_set].append(sample_dist)

# Converting all to Numpy arrays
for ood_set in ood_dists:
    ood_dists[ood_set] = np.array(ood_dists[ood_set])

# iNat 12 13
# Places 11 13
# SUN 12 13
# DTD 34 78

# Getting metrics such as AUROC, FPR at TPR 95
for ood_set in ood_dists:
    results = cal_metric(id_ious, ood_dists[ood_set])
    print(f"FPR for {ood_set} is {results['FPR']*100}")
    print(f"AUROC for {ood_set} is {results['AUROC']*100}")
    plot_distrib(id_dist, ood_dists[ood_set], path='broden/results/distrib/{out_dataset}_{method}.png'.format(method='Euclidean',out_dataset=ood_set),
                      title="{out_dataset} {method}".format(method='Euclidean Distance', out_dataset=ood_set))
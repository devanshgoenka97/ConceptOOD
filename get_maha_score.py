from __future__ import print_function

import numpy as np
import pickle
import os
from util.metrics import cal_metric, plot_distrib
import math
import csv

np.random.seed(1)

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

# Fetching stored train ID scores
with open('broden/results/id_score_train.pkl', 'rb') as f:
    in_scores = pickle.load(f)

maha_params = {}
# For each target in ID dataset
for target in in_scores:
    # For each sample in target - 1000 for each target in ImageNet training set
    sample_array = []
    for index in range(1000):
        sample_probs = []
        # For each concept in target
        for concept in in_scores[target]:
            try:
                sample_probs.append(np.around(in_scores[target][concept][index]))
            except IndexError:
                # Need to handle uneven class size case
                break
        
        # If any probability scores extracted then append to sample array 
        if len(sample_probs) > 0:
                sample_array.append(np.array(sample_probs))

    # Estimating the sample mean and population covariance to model each target as a Gaussian
    sample_array = np.array(sample_array)
    sample_mean = np.mean(sample_array, axis=0)
    sample_covI = np.linalg.inv(np.cov(sample_array.T) + 10e-6 * np.eye(sample_mean.shape[0]))
    maha_params[target] = {'mean': sample_mean, 'cov': sample_covI}
    
# Now we have sample mean and sample covariance for all targets in ID dataset, we can perform retrieve mahalanobis score as follows:
# We will forward propagate our test set through the model to get the binary vector for that sample
# Then we will get the mahalanobis score of the sample against all the gaussians given by  each target
# Final mahalanobis score will be the maximum among all mahalanobis scores, and then if maha_min > threshold, it is OOD else ID

def max_maha_score(sample, maha_params):
    max_maha = -1e+10
    # Checking against each target set
    for target in maha_params:
        # Sample mean
        mean = maha_params[target]['mean']
        # Sample covariance
        covI = maha_params[target]['cov']
        # Mahalanobis distance
        maha = -math.sqrt(np.dot(np.dot(covI, sample-mean), (sample-mean).T))
        max_maha = max(max_maha, maha)
    return max_maha

# This part propagates each test sample to get the Mahalanobis scores for each image in ID dataset.

# Fetching stored ID scores
with open('broden/results/id_score_val.pkl', 'rb') as f:
    test_scores = pickle.load(f)

id_mahas = []

# For each target in ID dataset
for target in test_scores:
    # For each sample in test target - 50 for each target in ImageNet validation set
    for index in range(50):
        sample_probs = []

        # For each concept in target
        for concept in test_scores[target]:
            sample_probs.append(np.around(test_scores[target][concept][index]))
        
        sample_probs = np.array(sample_probs)
        sample_maha_score = max_maha_score(sample_probs, maha_params)
        id_mahas.append(sample_maha_score)

id_mahas.sort()
threshold = id_mahas[round(0.05 * len(id_mahas))]

id_mahas = np.array(id_mahas)

print(f"Threshold is {threshold}")

with open('broden/results/ood_scores_new.pkl', 'rb') as f:
    ood_scores = pickle.load(f)

ood_mahas = {}
for ood_set in ood_scores:
    ood_mahas[ood_set] = []

# For each OOD dataset
for ood_set in ood_scores:
    # For each image in OOD dataset
    n = len(ood_scores[ood_set][true_concepts[0]])
    for index in range(n):
        probs = []
        # For each concept score of the image
        for concept in ood_scores[ood_set]:
            probs.append(np.around(ood_scores[ood_set][concept][index]))

        probs = np.array(probs)
        sample_maha_score = max_maha_score(probs, maha_params)
        ood_mahas[ood_set].append(sample_maha_score)

# Converting all to Numpy arrays
for ood_set in ood_mahas:
    ood_mahas[ood_set] = np.array(ood_mahas[ood_set])

# Getting FPR at TPR 95 and AUROC
for ood_set in ood_mahas:
    results = cal_metric(id_mahas, ood_mahas[ood_set])
    print(f"FPR for {ood_set} is {results['FPR']*100}")
    print(f"AUROC for {ood_set} is {results['AUROC']*100}")
    if ood_set == 'inat':
        name = 'iNaturalist'
    elif ood_set == 'sun':
        name = 'SUN'
    elif ood_set == 'places':
        name = 'Places'
    else:
        name = 'Textures'
    plot_distrib(id_mahas, ood_mahas[ood_set], path='broden/results/distrib/{out_dataset}_{method}.png'.format(method='Mahalanobis Distance',out_dataset=ood_set),
                      title="{out_dataset}".format(method='', out_dataset=name))
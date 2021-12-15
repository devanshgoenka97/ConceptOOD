from __future__ import print_function

import numpy as np
import pickle
import os
from util.metrics import cal_metric, plot_distrib
import torch
import csv

np.random.seed(1)

# Now we have k-closest concepts to all targets in ID dataset, we can perform retrieve IoU score as follows:
# We will forward propagate our test set through the model to get the k-closest concepts to that sample
# Then we will get the IoU score of the sample against all the k-closest concepts to each target
# Final IoU score will be the maximum among all IoU scores, and then if IoU_max > threshold, it is ID else OOD

def max_iou_score(sample, target_closest):
    max_iou= 0
    # Checking against each target set
    for concept in target_closest:
        # Intersection
        intersection = np.intersect1d(sample, target_closest[concept])
        # Union
        union = np.union1d(sample, target_closest[concept])
        # IoU score
        iou = intersection.shape[0] / union.shape[0]
        max_iou = max(max_iou, iou)
    return max_iou

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

k = 15

result = '''
<html>
    <body>
        <h1>IoU score results</h1>
        <h3>In-Distribution Closest Concepts</h3>
        <div> 
        {0}
        </div>
        <h3>Out-of-Distribution Closest Concepts</h3>
        <div> 
        {1}
        </div>
    </body>
</html>
'''

result_file = open('artifacts/results/iou.html', 'w')

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

# Getting k-closest concepts to each target in the ID dataset
closest_concepts = {}
for target in in_scores:
    closest_concepts[target] = true_concepts[np.argsort(-in_scores[target])[:k]]

id_out = ''' '''

for target in closest_concepts:
    toplabels = ','.join([labels[l] for l in closest_concepts[target]])
    id_out += 'Target: ' + str(target) + ', Closest Concepts : ' + toplabels + '<br />'

# This part propagates each test sample to get the IoU scores for each image in ID dataset.

# Fetching stored validation ID scores
with open('broden/results/id_score_val.pkl', 'rb') as f:
    test_scores = pickle.load(f)

id_ious = []

# For each target in ID dataset
for target in test_scores:
    # For each sample in test target - 50 for each target in ImageNet validation set
    for index in range(50):
        sample_probs = []

        # For each concept in target
        for concept in test_scores[target]:
            sample_probs.append(test_scores[target][concept][index])
        
        topk = torch.tensor(sample_probs).topk(k)
        sample_topk_concepts = true_concepts[topk.indices]
        sample_iou_score = max_iou_score(sample_topk_concepts, closest_concepts)
        id_ious.append(sample_iou_score)

id_ious.sort()
threshold = id_ious[round(0.05 * len(id_ious))]

id_ious = np.array(id_ious)

print(f"Threshold is {threshold}")
print(f"K is {k}")

with open('broden/results/ood_scores_new.pkl', 'rb') as f:
    ood_scores = pickle.load(f)

ood_ious = {}
for ood_set in ood_scores:
    ood_ious[ood_set] = []

ood_out = ''' '''

# For each OOD dataset
for ood_set in ood_scores:
    # For each image in OOD dataset
    n = len(ood_scores[ood_set][true_concepts[0]])
    for index in range(n):
        probs = []

        # For each concept score of the image
        for concept in ood_scores[ood_set]:
            probs.append(ood_scores[ood_set][concept][index])

        # Getting top k probs for each image in OOD set
        topk = torch.tensor(probs).topk(k)
        topconcepts = true_concepts[topk.indices]
        toplabels = [labels[l] for l in topconcepts]
        sample_iou_score = max_iou_score(topconcepts, closest_concepts)
        if sample_iou_score > threshold:
            ood_out += 'OOD Dataset: ' + ood_set + ', Index : ' + str(index) + ', Top Concepts: ' + ','.join(toplabels) + '<br/>'
        ood_ious[ood_set].append(sample_iou_score)

# Converting all to Numpy arrays
for ood_set in ood_ious:
    ood_ious[ood_set] = np.array(ood_ious[ood_set])

result = result.format(id_out, ood_out)
result_file.write(result)
result_file.close()

# Getting metrics such as AUROC, FPR at TPR 95
for ood_set in ood_ious:
    results = cal_metric(id_ious, ood_ious[ood_set])
    print(f"FPR for {ood_set} is {results['FPR']*100}")
    print(f"AUROC for {ood_set} is {results['AUROC']*100}")
    plot_distrib(id_ious, ood_ious[ood_set], path='artifacts/results/distrib/{out_dataset}_{method}_{param}.png'.format(method='IoU',out_dataset=ood_set, param=k),
                      title="{out_dataset}".format(method='', out_dataset=ood_set))
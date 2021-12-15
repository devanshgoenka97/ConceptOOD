from __future__ import print_function
import os

import torch
import torch.nn as nn
import numpy as np
from util.args_loader import get_args
from util.data_loader import get_loader_probe, num_classes_dict
import numpy as np
import csv
import pickle

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def broden_indice_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    tensor_batch = [d['tensor'] for d in batch]
    indices_batch = [d['i'] for d in batch]
    tensors = torch.stack(tensor_batch, 0)
    return tensors, indices_batch

args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

args.in_dataset = 'imagenet'
args.batch_size = 64

loader_probe = get_loader_probe(args)
dataset = loader_probe.dataset
num_classes = num_classes_dict['imagenet']


all_indices = list(range(dataset.__len__()))

activation_map = {}

print("Loading activations from memory")

# Loading stored pre-trained activations into memory
with open('artifacts/activations.pkl', 'rb') as f:
    activation_map = pickle.load(f)

print("Loaded all activations")

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

accuracies = {}

for i, concept in enumerate(concepts):
    # Attaching linear classifier
    layer = nn.Linear(512, 1).cuda()

    if not os.path.exists("artifacts/classifiers/concept_classifier_{0}.pth".format(concept)):
        print("Classifier was not trained due to too few samples")
        continue

    checkpoint = torch.load("artifacts/classifiers/concept_classifier_{0}.pth".format(concept))
    layer.load_state_dict(checkpoint['model_state_dict'])
    layer.eval()
    
    print(f"Testing for concept : {concept}")
    
    # Loading indices of all positive samples from broden
    with open("artifacts/pos_samples/concept_pos_{0}.pkl".format(concept), "rb") as f:
        pos_indices = pickle.load(f)
    
    # Fetching negative indices by intersection of all indices and positive indices
    neg_indices = list(set(all_indices) - set(pos_indices))

    # Creating separate train loaders for positive and negative samples
    trainset_pos = torch.utils.data.Subset(dataset, pos_indices)
    trainset_neg = torch.utils.data.Subset(dataset, neg_indices)

    trainloader_pos = torch.utils.data.DataLoader(trainset_pos, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=broden_indice_collate)
    trainloader_neg = torch.utils.data.DataLoader(trainset_neg, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=broden_indice_collate)

    with torch.no_grad():
        total = 0
        correct = 0
        for j, data in enumerate(zip(trainloader_pos, trainloader_neg)):
            pos, neg = data

            inputs_pos, indices_pos = pos
            inputs_neg, indices_neg = neg

            inputs = torch.cat((inputs_pos, inputs_neg), dim=0).cuda()

            # Creating binary labels for input batch
            ones = torch.ones(inputs_pos.shape[0]).cuda()
            zeros = torch.zeros(inputs_neg.shape[0]).cuda()
            labels = torch.cat((ones, zeros), dim=0)

            curr_batch_size = inputs.shape[0]

            # Combining the broden indices to form a single array
            indices = np.concatenate((indices_pos, indices_neg))
            
            # Fetching activations for the current indices
            activations = np.array([np.array(activation_map[index]) for index in indices])
            activations = torch.tensor(activations).cuda()

            curr_batch_size = inputs.shape[0]
            total += curr_batch_size

            # Using probed network activations to pass through the linear classifier
            outputs = layer(activations).squeeze()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (labels == predicted).sum().float()
            if (j+1) % 100 == 0:
                print(f"[Batch {j+1}]: Accuracy : {correct/total}")

    accuracies[concept] = (correct/total).item()
    print(f"Accuracy : {correct/total}")
    print(f"Done testing for concept {concept}")

print("Found all accuracies")
save_dir = os.path.join('artifacts', 'results')
os.makedirs(save_dir, exist_ok=True)
save_file = "accuracies.pkl"
with open(os.path.join(save_dir, save_file), 'wb') as f:
    pickle.dump(accuracies, f)
print("Storing accuracies")
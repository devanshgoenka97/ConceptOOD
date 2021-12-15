from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from util.args_loader import get_args
from util.data_loader import get_loader_probe, num_classes_dict
import numpy as np
import pickle
import csv

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

# Using BCE with logits for better numerical stability
criterion = nn.BCEWithLogitsLoss().cuda()

# All indices of images in the Broden dataset
all_indices = list(range(dataset.__len__()))

activation_map = {}

print("Loading activations from memory")

# Loading stored pre-trained activations into memory
with open('broden/activations.pkl', 'rb') as f:
    activation_map = pickle.load(f)

# Only reading the object file to extract semantics
concepts = []
with open('/nobackup/broden1_224/c_part.csv', 'r') as f:
    csvreader = csv.reader(f)
    # Elegantly ignoring first row in CSV file
    skip  = 0
    while(skip >= 0):
        next(csvreader)
        skip -= 1
    for row in csvreader:
        concepts.append(int(row[1]))

print("Loaded all activations")

for i, concept in enumerate(concepts):
    # Checking if classifier already exists
    if os.path.exists("broden/classifiers_2/concept_classifier_{0}.pth".format(concept)):
        continue

    # Training a linear classifier for each concept
    layer = nn.Linear(512, 1).cuda()
    layer.train()

    optimizer = optim.SGD(layer.parameters(), lr=0.001, momentum=0.9)

    print(f"Training for concept : {concept}")

    # Loading indices of all positive samples from broden
    with open("broden/pos_samples/concept_pos_{0}.pkl".format(concept), "rb") as f:
        pos_indices = pickle.load(f)

    # Skipping concepts with very few positive examples
    if len(pos_indices) < 50:
        print(f"Too few examples for concept, skipping concept {concept}")
        continue
    
    # Fetching negative indices by intersection of all indices and positive indices
    neg_indices = list(set(all_indices) - set(pos_indices))

    # Creating separate train loaders for positive and negative samples
    trainset_pos = torch.utils.data.Subset(dataset, pos_indices)
    trainset_neg = torch.utils.data.Subset(dataset, neg_indices)

    trainloader_pos = torch.utils.data.DataLoader(trainset_pos, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=broden_indice_collate)
    trainloader_neg = torch.utils.data.DataLoader(trainset_neg, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=broden_indice_collate)

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        import time
        start = time.time()
        for j, data in enumerate(zip(trainloader_pos, trainloader_neg)):
            # get the inputs; data is a list of [inputs, labels]
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
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Using probed network activations to pass through the linear classifier
            outputs = layer(activations).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if (j+1) % 10 == 0:    # print every 10 mini-batches
                print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, j + 1, running_loss/10))
                print('Time taken : %.3f' % (time.time() - start))
                start = time.time()
                running_loss = 0.0

    print(f"Done training for concept : {concept}")
    CONCEPT_PATH = "artifacts/classifiers/concept_classifier_{0}.pth".format(concept)
    torch.save({
            'epoch': 20,
            'model_state_dict': layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, CONCEPT_PATH)
    print(f"Saved state for concept : {concept}")
    

print('Finished Training')
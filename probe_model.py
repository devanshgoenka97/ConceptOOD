from __future__ import print_function
import os

from models.resnet import resnet18

import torch
import torch.nn.functional as F
import numpy as np
from util.args_loader import get_args
from util.data_loader import get_loader_probe, num_classes_dict
import numpy as np
import pickle


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args.in_dataset = 'imagenet'
args.batch_size = 64
loader_probe = get_loader_probe(args)
dataset = loader_probe.dataset
num_classes = num_classes_dict['imagenet']

# Concept ResNet based on pre-trained ImageNet
model = resnet18(pretrained=True)

model = model.cuda()

activations = {}

with torch.no_grad():
    for j, data in enumerate(loader_probe):
        inputs, info_dict = data
        inputs = inputs.cuda()
        curr_batch_size = inputs.shape[0]
        outputs = F.avg_pool2d(model.features(inputs), 7).view(inputs.shape[0], 512)
        for i in range(curr_batch_size):
            activations[info_dict[i]['i']] = outputs[i].detach().cpu().numpy()

        if (j+1) % 100 == 0:    # print every 100 mini-batches
            print(f"Done storing for batch :{j+1}")

f = open('artifacts/activations.pkl', 'wb')
pickle.dump(activations, f)

print('Finished extracting all model activations')
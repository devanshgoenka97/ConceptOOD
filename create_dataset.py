from __future__ import print_function
import os
import torch
import numpy as np
from util.args_loader import get_args
from util.data_loader import get_loader_probe, num_classes_dict
import numpy as np
import pickle

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.in_dataset = 'imagenet'
args.batch_size = 1000
loader_probe = get_loader_probe(args)
dataset = loader_probe.dataset
num_classes = num_classes_dict['imagenet']

def is_present(i, truth_label, info_dict):
    for cate in dataset.categories:
        # segmentation mask for the given category
        segmentation = info_dict[i][cate]
        if type(segmentation) == np.ndarray and segmentation.max() == 0: continue
        if type(segmentation) == list and len(segmentation) == 0: continue
        if type(segmentation) == np.ndarray:
            # reshape the segmentation mask to the feature map's size to be aligned
            for seg_id in range(len(segmentation)):
                seg_2d = segmentation[seg_id]
                # for each concept label in the segmentation mask, collect the feature
                for label in np.unique(seg_2d):
                    if label == truth_label:
                        return True
                    # Add to label

            # if the segmentation is a list of label, then it means the entire image is marked with the given labels.
        elif type(segmentation) == list:
            for label in segmentation:
                if label == truth_label:
                    return True
                # Add to labels


for i in range(1, 1198):
    concept = i
    indices = []
    for j, data in enumerate(loader_probe):
        inputs, info_dict = data
        batch_size = inputs.shape[0]
        for k in range(batch_size):
            if is_present(k, concept, info_dict):
                indices.append(info_dict[k]['i'])
    indices = np.array(indices)
    save_dir = os.path.join('broden', 'pos_samples')
    os.makedirs(save_dir, exist_ok=True)
    save_file = "concept_pos_{0}.pkl".format(concept)
    print(f"Storing positive samples for Concept: {concept}")
    with open(os.path.join(save_dir, save_file), 'wb') as f:
        pickle.dump(indices, f)


print('Finished Storing Positive Samples')
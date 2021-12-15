from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import time
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
from util.score import get_score


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def eval_ood_detector(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    batch_size = args.batch_size
    method = args.method
    method_args = args.method_args
    name = args.name

    in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat')
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split=('val'))
    testloaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    # loader_val_dict = get_loader_out(args, dataset=(None, args.val_dataset), split=('val'))
    # valloaderOut = loader_val_dict.val_ood_loader
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)
    print()

    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")
        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                logits = model(inputs)

                outputs = F.softmax(logits, dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

                for k in range(preds.shape[0]):
                    g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            scores = get_score(inputs, model, method, method_args, logits=logits)
            for score in scores:
                f1.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f1.close()
        g1.close()

    if mode_args['in_dist_only']:
        return

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                logits = model(inputs)

            scores = get_score(inputs, model, method, method_args, logits=logits)
            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()
        f2.close()

    return

if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()

    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only

    if args.method == "odin":
        args.method_args['temperature'] = 1000.0
        param_dict = {
            "CIFAR-10": {
                "resnet18": 0.01,
                "resnet18_cl0.8": 0.01,
                "resnet18_cl1.0": 0.07,
            },
            "CIFAR-100": {
                "resnet18": 0.04,
                "resnet18_cl1.0": 0.04,
            },
            "imagenet":{
                "resnet50": 0.005,
                "resnet50_cl1.0": 0.0,
                "mobilenet": 0.03,
                "mobilenet_cl1.3": 0.04,
            }
        }
        args.method_args['magnitude'] = param_dict[args.in_dataset][args.name]
    if args.method == "godin":
        param_dict = {
            "CIFAR-10": {
                "densenet_godin": 0.02,
                "resnet18_godin": 0.02,
            },
            "CIFAR-100": {
                "densenet_godin_full": 0.02,
                "resnet18_godin": 0.02,
            },
            "imagenet": {
                "resnet50-godin": 0.03,
                "mobilenet-godin": 0.02,
            }
        }
        args.method_args['magnitude'] = param_dict[args.in_dataset][args.name]
    elif args.method == 'mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'), allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        args.method_args['sample_mean'] = sample_mean
        args.method_args['precision'] = precision
        args.method_args['magnitude'] = magnitude
        args.method_args['regressor'] = regressor
        args.method_args['num_output'] = 4
    elif args.method == 'modedmaha':
        # import pickle
        # cache_path = os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'tmp', "estimate_cache.pickle")
        # with open(cache_path, 'rb') as f:
        #     sample_mean, precision, precision_class = pickle.load(f)
        sample_mean, precision, precision_class, lr_weights, lr_bias, magnitude = np.load(
            os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'),
            allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        args.method_args['sample_mean'] = sample_mean
        args.method_args['precision'] = precision
        args.method_args['precision_class'] = precision_class
        args.method_args['magnitude'] = 0.0
        args.method_args['regressor'] = regressor
        args.method_args['layer_indices'] = [3]
        args.method_args['p'] = args.p
    elif args.method == 'energy':
        args.method_args['temperature'] = 1000.0
    elif args.method == 'gram':
        Mins, Maxs = np.load("output/ood_scores/{}/{}/minmax.npy".format(args.in_dataset, args.name))
        Eva = np.load("output/ood_scores/{}/{}/eva.npy".format(args.in_dataset, args.name))
        args.method_args['Mins'] = Mins
        args.method_args['Maxs'] = Maxs
        args.method_args['Eva'] = Eva
    elif args.method == 'gradient':
        args.method_args['temperature'] = 1
        gradient_cls_mean = np.load(os.path.join('output/gradient_stat/', args.in_dataset, args.name, 'gradient_cls_mean.npy'))
        args.method_args['gradient_cls_mean'] = gradient_cls_mean
        args.method_args['p'] = args.p

        grad_mask = torch.ones(gradient_cls_mean.shape)
        if args.p > 0:
            for cls in range(gradient_cls_mean.shape[0]):
                v = gradient_cls_mean[cls]
                grad_mask[cls] = torch.from_numpy(v > np.percentile(v, args.p))
        args.method_args['grad_mask'] = grad_mask

    eval_ood_detector(args, mode_args)
    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.method, args.name)
    compute_in(args.base_dir, args.in_dataset, args.method, args.name)

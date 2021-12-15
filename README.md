# On Using Concept Disentanglement for Out-of-Distribution Detection

This is the source code for the project done in Fall 2021 by Devansh Goenka and Yixuan Li.

In this work, we propose ConceptOOD, a technique for introducing interpretability while performing OOD detection.
Our method is motivated by the findings in [Network Dissection](http://netdissect.csail.mit.edu/final-network-dissection.pdf) and [Net2Vec](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fong_Net2Vec_Quantifying_and_CVPR_2018_paper.pdf) which have found that convolutional neural networks can learn disentangled concepts, which we leverage to perform OOD detection along with the potential to explain these predictions. 

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download the Broden dataset for probing the pre-trained network from [here](https://github.com/CSAILVision/NetDissect/blob/release1/script/dlbroden.sh).

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/id_data/ILSVRC-2012/train` and  `./datasets/id_data/ILSVRC-2012/val`, respectively.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data/`.

### 2. Pre-trained Model Preparation

The model we used in this work is the ResNet-18 pretrained on ImageNet as provided by PyTorch.

### 3. OOD Detection Evaluation

To reproduce our results on ResNet-18 please perform the following steps:

* First, we segregate Broden examples for each concept into positive and negative sets
```
python create_dataset.py
```

* Record activations of pre-trained network on all Broden images to speed up training.
```
python probe_model.py
```

* Learn concept classifiers for each concept.
```
python learn_concepts.py
```

* Score training data to estimate class-conditional parameters for scoring functions.
```
python id_score.py
```

* Score OOD data to record concept probabilities for each sample.
```
python ood_score.py
```

* Perform scoring with IoU method.
```
python get_iou_score.py
```

* Perform scoring with Mahalanobis method.
```
python get_maha_score.py
```


## OOD Detection Results

ConceptOOD achieves state-of-the-art performance in terms of FPR95 averaged on the 4 OOD datasets.

![results](figs/results.png)
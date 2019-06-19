"""
flips labels for simulating label noise experiments with non-uniform label noise
"""

import numpy as np
import cPickle as cp
import argparse
import random

import pdb


parser = argparse.ArgumentParser(description='label flipping script for various data sets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--eta', default=0., type=float, help='label flipping probability')
parser.add_argument('--dataset', required=True, type=str, help='dataset to use. cifar10/cifar100/fashion')
parser.add_argument('--labels', required=True, type=str, help='label file containing original labels. Must be a numpy array')
parser.add_argument('--outfile', default="label_flip_out", type=str, help='output pickle file to store flipped indices and new labels')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')

args = parser.parse_args()

np.random.seed(args.seed)

def flip_label():
	return np.random.uniform() < args.eta


##########
#CIFAR-10
##########
cifar10_labels = ['airplane',
	'automobile',
	'bird',
	'cat',
	'deer',
	'dog' ,
	'frog',
	'horse',
	'ship',
	'truck']

#src_classes get flipped to corresponding  target classes with some probability
cifar10_flip_src_class = ['truck','bird','deer','cat','dog']
cifar10_flip_target_class = ['automobile','airplane','horse','dog','cat']



###########
#CIFAR-100
###########
#each line is a group of five related classes. 
#see https://www.cs.toronto.edu/~kriz/cifar.html
cifar100_labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'containers	bottles', 'bowls', 'cans', 'cups', 'plates',
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple', 'oak', 'palm', 'pine', 'willow',
'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
'lawn-mower', 'rocket','streetcar', 'tank', 'tractor']


cifar100_flip_src_class = cifar100_labels
#circular mapping within cifar-100 groups. each line (see cifar 100 class names above) is a related group
#of objects.
cifar100_flip_target_class = [cifar100_flip_src_class[i/5*5 + (i+1)%5] for i in range(len(cifar100_flip_src_class))]
#pdb.set_trace()


################
# Fashion-MNIST
################
fashion_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boot']
#src_class objects get probabilistically flipped to corresponding target class objects
fashion_flip_src_class = ['boot','sneaker','pullover','coat']
fashion_flip_target_class = ['sneaker','sandal','shirt','dress']


if args.dataset == 'cifar10':
	labels = cifar10_labels
	src_class = cifar10_flip_src_class
	target_class = cifar10_flip_target_class

elif args.dataset == 'cifar100':
	labels  = cifar100_labels
	src_class = cifar100_flip_src_class
	target_class = cifar100_flip_target_class

elif args.dataset == 'fashion':
	labels = fashion_labels
	src_class = fashion_flip_src_class
	target_class = fashion_flip_target_class

else:
	print "unsupported data set"
	exit()



y_orig = np.load(args.labels)
flipped_indices = []
flipped_labels = []

for idx,l in enumerate(y_orig):
	class_name = labels[l]
	if  class_name in src_class:
		if flip_label():
			src_index = src_class.index(class_name)
			new_class = target_class[src_index]
			new_label = labels.index(new_class)
			flipped_indices.append(idx)
			flipped_labels.append(new_label)
		#else, do nothing
	#else, do nothing

flipped_indices = np.asarray(flipped_indices,dtype=np.int64)
flipped_labels = np.asarray(flipped_labels, dtype=np.int64)

print("flipped %d indices" %(len(flipped_indices)))
print("pickling to %s" %(args.outfile+".pkl"))
cp.dump((flipped_indices,flipped_labels),open(args.outfile+".pkl",'wb'))

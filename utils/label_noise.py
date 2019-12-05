import numpy as np
import pdb
try:
	import cPickle as cp
except ModuleNotFoundError: #no cPickle in python 3
	import pickle as cp

def label_noise(args, trainset, num_classes):

	if args.rand_labels is not None:
		#pdb.set_trace()
		rf = args.rand_labels
		if rf < 0. or rf > 1.:
			print("rand_labels fraction should be between 0 and 1")
			sys.exit(0)
		#pdb.set_trace()
		if args.dataset == 'stl10-labeled' or args.dataset=='tin200':
			train_labels = np.asarray(trainset.labels)
		else:
			#train_labels = np.asarray(trainset.train_labels)
			train_labels = np.asarray(trainset.targets)

		print("randomizing %f percent of labels " %(rf*100))
		n_train = len(train_labels)
		n_rand = int(rf*n_train)
		randomize_indices = np.random.choice(range(n_train),size=n_rand,replace=False)

		#pdb.set_trace()

		train_labels_old = np.copy(train_labels)

		train_labels[randomize_indices] = np.random.choice(range(num_classes),size=n_rand,replace=True)

		wrong_indices = np.where(train_labels != train_labels_old)[0]
		wrong_labels = train_labels[wrong_indices]

		#pdb.set_trace()


		if args.del_noisy_data:
			print("deleting noisy data")
			#trainset.train_data = np.delete(trainset.train_data,wrong_indices,axis=0)
			trainset.data = np.delete(trainset.data,wrong_indices,axis=0)
			train_labels = np.delete(train_labels,wrong_indices)

		#pdb.set_trace()
		if args.dataset == 'stl10-labeled' or args.dataset=='tin200':
			trainset.labels = train_labels.tolist()
		#elif args.dataset == 'mnist':
		#	trainset.targets = train_labels.tolist()
		else:
			#trainset.train_labels = train_labels.tolist()
			trainset.targets = train_labels.tolist()

		print("training on %d data samples" %(len(trainset.data)))

		#save randomized indices if validation or train scores are also being saved
		if args.save_val_scores or args.save_train_scores:
			if not args.log_file is None:
				fn = args.log_file.replace(".log","")
			else:
				fn = 'test' #assuming that if a log file has not been specified this is a test run.
			#print("saving randomized indices to %s" %(fn+"_rand_indices.npy"))
			print("saving wrong (of randomized) indices and labels to %s" %(fn+"_corrupt_label_info.pkl"))
			#np.save(fn+"_rand_indices", wrong_indices)
			cp.dump((wrong_indices,wrong_labels),open(fn+"_corrupt_label_info.pkl",'wb'))
		
		#pdb.set_trace()

	train_labels_good = None
	#pdb.set_trace()
	if args.label_noise_info is not None:
		if args.rand_labels is not None:
			print("rand_labels option cannot be chosen when --label_noise_info is active")
			quit()
		print("using label noise info specified in ", args.label_noise_info)
		try:
			(noise_indices, noise_labels) = cp.load(open(args.label_noise_info,'rb'))
		except UnicodeDecodeError: #reading python 2 pickles in python 3 can throw this error. 
			(noise_indices, noise_labels) = cp.load(open(args.label_noise_info,'rb'),encoding='bytes')

		if args.dataset == 'stl10-labeled' or args.dataset == 'stl10-c' or args.dataset=='tin200':
			train_labels = np.asarray(trainset.labels)
			train_labels_good = np.copy(train_labels)
			train_labels[noise_indices] = noise_labels
			trainset.labels = train_labels.tolist()
		else: # cifar-10/100 and others
			train_labels = np.asarray(trainset.targets)
			train_labels_good = np.copy(train_labels)
			train_labels[noise_indices] = noise_labels
			#trainset.train_labels = train_labels.tolist()
			trainset.targets = train_labels.tolist()

	if args.exclude_train_indices is not None:
		if args.rand_labels is not None:
			print("rand_labels option cannot be chosen when excluding train indices")
			quit()
		print("excluding training indices specified in ", args.exclude_train_indices)
		exclude_indices = np.load(args.exclude_train_indices)
		#pdb.set_trace()
		if args.dataset == 'stl10-c':
			trainset.data = np.delete(trainset.data, exclude_indices,axis=0)
		else:
			trainset.data = np.delete(trainset.data, exclude_indices,axis=0)
		if args.dataset == 'stl10-labeled' or args.dataset == 'stl10-c' or args.dataset=='tin200':
			train_labels = np.asarray(trainset.labels)
			train_labels = np.delete(train_labels,exclude_indices)
			trainset.labels = train_labels.tolist()
		else: # cifar-10/100 and others
			train_labels = np.asarray(trainset.targets)
			train_labels = np.delete(train_labels,exclude_indices)
			#trainset.train_labels = train_labels.tolist()
			trainset.targets = train_labels.tolist()

		if args.label_noise_info is not None:
			assert(train_labels_good is not None)
			train_labels_good = np.delete(train_labels_good, exclude_indices)


	if args.label_noise_info is not None:
		assert(len(train_labels) == len(train_labels_good))
		label_noise_ratio  = float(len(np.where(train_labels != train_labels_good)[0]))/len(train_labels_good)
		print("Trainset size %d. Label Noise ratio %f" %(len(train_labels), label_noise_ratio))


	return trainset

import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings 
warnings.filterwarnings('ignore')

from snorkel.labeling.model import LabelModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

import os.path

sns.set_style("darkgrid")
# import sys
# np.set_printoptions(threshold=sys.maxsize)

from lpa.utils import GenerateMatrix, KhopNeighbor, Acc, normalize_matrix, AdjustAcc, NotAbstainAcc
from lpa.label_prop import PropagationSoft, PropagationHard, PropagationAdaptive, smoothing_wl, compute_prior_reg, compute_step_prior
from sklearn.metrics import pairwise_distances
import lpa.dataset
import argparse

def get_results(soft_preds, labels, con_idx, method_name, split):

	'''
	Helper function that wraps generation of results into a resulting list (fed into Pandas)

	Args:
	soft_preds - outputs of a LPA scheme
	con_idx - indices representing data connected to a labeled point
	method_name - string representing LPA scheme
	split - which split of the data
	'''

	return [method_name, AdjustAcc(soft_preds, labels),  \
			NotAbstainAcc(soft_preds, labels), (np.abs(soft_preds[:,1] - 0.5) > 0.001).sum() / labels.shape[0],
			Acc(soft_preds[con_idx], labels[con_idx]), con_idx.shape[0] / labels.shape[0], split]


def load_data(data_name, euc_th, wl_th, seed):

	'''
	Function to a saved split of data + graph information

	Args:
	data_name - string representing the dataset
	euc_th - value representing the thresholding on the euclidean radius graph
	wl_th - value representing the threshold on a graph to smooth weak labelers (for ALPA)
	seed - random seed number
	'''


	# Load raw data
	base_path = "datasets/" + data_name + "/"
	if os.path.isfile(base_path + "/train_X_seed" + str(seed)): 
		train_X = torch.load(base_path + "/train_X_seed" + str(seed))
		train_L = torch.load(base_path + "/train_L_seed" + str(seed))
		train_labels = torch.load(base_path + "/train_labels_seed" + str(seed))
	else:
		train_dataset = dataset.WSDataset(dataset=data_name, split="train", feature="bert", balance=True, seed = seed)
		train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

		train_X, train_L, train_labels = next(iter(train_loader))
		train_X, train_L = train_X.type(torch.float), train_L.type(torch.long)
		torch.save(train_X, base_path + "/train_X_seed" + str(seed))
		torch.save(train_L, base_path + "/train_L_seed" + str(seed))
		torch.save(train_labels, base_path + "/train_labels_seed" + str(seed))
		
	X = train_X
	L = train_L
	labels = train_labels

	# Load adjacency matrix
	euc_mat = pairwise_distances(X, Y = None, metric='euclidean') 

	base_path = "datasets/" + data_name + "/"
	if os.path.isfile(base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(euc_th)): 
		W_x = torch.load(base_path + "/W_x_seed" + str(seed) + "_thresh_" + str(euc_th))
		S_x = torch.load(base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(euc_th))
	else:
		W_x, S_x = GenerateMatrix(euc_mat, thresh = euc_th)
		torch.save(W_x, base_path + "/W_x_seed" + str(seed) + "_thresh_" + str(euc_th))
		torch.save(S_x, base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(euc_th))
	
	if os.path.isfile(base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(wl_th)): 
		W_x_large = torch.load(base_path + "/W_x_seed" + str(seed) + "_thresh_" + str(wl_th))
		S_x_large = torch.load(base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(wl_th))
	else:
		W_x_large, S_x_large = GenerateMatrix(euc_mat, thresh = wl_th)
		torch.save(W_x_large, base_path + "/W_x_seed" + str(seed) + "_thresh_" + str(wl_th))
		torch.save(S_x_large, base_path + "/S_x_seed" + str(seed) + "_thresh_" + str(wl_th))

	return X, L, labels, W_x, S_x, W_x_large, S_x_large

def get_auggraph(L, labels, S_x_large, a, seed):

	'''
	Function to generate an augmented graph used in ALPA (and returning base predictions)

	Args:
	L - weak labeler votes
	labels - small set of true labels used for propagations
	wl_smoothing_graph = graph used to smooth weak labelers for ALPA
	a - threshold to prune edges
	'''


	# Weak labels aggregation
	label_model = LabelModel(cardinality=2, verbose=False)
	label_model.fit(L_train=L, n_epochs=200, log_freq=200, seed=seed)
	pseudolabs_soft = label_model.predict_proba(L=L)
	base_preds = pseudolabs_soft

	# Smooth weak labels
	smooth_wl = PropagationSoft(base_preds, S_x_large , labels, labeled_inds = [], alpha = 10)
	abs_th = np.quantile(np.max(smooth_wl , axis = 1), a)

	# Construct a new graph from wl partitions
	sim_threshold = 0.01
	L_tf = (-1)* (L == 0) + (L == 1)
	cos_mat = (pairwise_distances(L_tf, Y=None, metric='cosine') < sim_threshold)*1.0
	W_wl = cos_mat.copy()

	# Remove points that weak labels are not consistent, abstain.
	# Confident wl
	unconwl_idx = np.max(smooth_wl , axis = 1) < abs_th
	unconwl_idx_array = np.array([i for i in range(len(unconwl_idx)) if unconwl_idx[i] == True])
	# Abstain weak labels
	abstain_wl_idx = (L.sum(axis=1) == -1 * L.shape[1])
	abstain_wl_idx_array = np.array([i for i in range(len(abstain_wl_idx)) if abstain_wl_idx[i] == True])

	if abstain_wl_idx_array != []:
		W_wl[abstain_wl_idx_array,:] = 0
		W_wl[:,abstain_wl_idx_array] = 0
	if unconwl_idx_array != []:
		W_wl[unconwl_idx_array,:] = 0
		W_wl[:,unconwl_idx_array] = 0

	return W_wl, base_preds

def no_smooth_experiment(data_name = 'youtube',  num_labels = 100, euc_th = 10, wl_th = 100, a = 0.6, mu = 0.01, lamb = 0.001):

	'''
	Wrapper script to run experiments given a dataset (without smoothing) and set of hyperparameters

	Args:
	data_name - dataset
	num_labels - amount of labeled data (for all of our experiments we use 100)
	euc_th - threshold for euclidean graph for LPA
	wl_th - threshold for smoothing graph for ALPA
	a - threshold for pruning edges in augmented graph
	mu - regularization weight
	lamb - 
	'''


	results = []
	seeds = range(5)
	
	for seed in seeds:
		
		### Load data
		np.random.seed(seed)
		labeled_inds = np.random.choice(range(len(X)), size=num_labels, replace=False)

		X, L, labels, W_x, S_x, W_x_large, S_x_large = load_data(data_name = data_name, euc_th = euc_th, wl_th = wl_th, seed = seed)
		W_wl, base_preds = get_auggraph(L, labels, S_x_large, a, seed)
		W_x_wl = W_x + mu * W_wl
		S_x_wl = normalize_matrix(W_x_wl)

		### Label propagation
		con_idx  = KhopNeighbor(labeled_inds, S_x, 1000) # get all connected
		con_idx2 = KhopNeighbor(labeled_inds, S_x_wl, 1000) # get all connected 
		alpha = 1/lamb

		# WL
		base_preds_wlb = base_preds[0].copy()
		base_preds_wlb[labeled_inds,:] = np.stack((1-labels[labeled_inds], labels[labeled_inds]), axis = 1)
		results.append(['wl', AdjustAcc(base_preds_wlb, labels), \
						NotAbstainAcc(base_preds_wlb, labels), \
						(base_preds_wlb[:,1]!= 0.5).sum() / labels.shape[0], 0, 0, "train"])
		# LP
		f_baseline = PropagationHard(np.ones_like(base_preds)*0.5, S_x, labels, labeled_inds, alpha = alpha)
		results.append(get_results(f_baseline, labels, con_idx, "lp", "train"))
		# LP + WL
		f_lp_wl = PropagationHard(base_preds, S_x, labels, labeled_inds, alpha = alpha)
		results.append(get_results(f_lp_wl, labels, con_idx, "lp+wl", "train"))
		# LPAG
		f_lp_ag = PropagationHard(np.ones_like(base_preds)*0.5, S_x_wl, labels, labeled_inds, alpha = alpha)
		results.append(get_results(f_lp_ag, labels, con_idx2, "lp+ag", "train"))
		# LPAG + WL
		f_lp_ag_wl = PropagationHard(base_preds, S_x_wl, labels, labeled_inds, alpha = alpha)
		results.append(get_results(f_lp_ag_wl, labels, con_idx2, "lp+ag+wl", "train"))
	return results

def gen_pl(data_name = 'youtube',  num_labels = 100, euc_th = 10, wl_th = 100, a = 0.6, mu = 0.01, lamb = 0.001, step=False, normalize=False):

	'''
	Wrapper script to run experiments given a dataset (without smoothing) and set of hyperparameters
	-> *** saves outputs of LPA and ALPA to use to train end_models (must run this before end_model.py)

	Args:
	data_name - dataset
	num_labels - amount of labeled data (for all of our experiments we use 100)
	euc_th - threshold for euclidean graph for LPA
	wl_th - threshold for smoothing graph for ALPA
	a - threshold for pruning edges in augmented graph
	mu - regularization weight
	lamb - 
	'''

	### LOAD DATA
	results = []
	seeds = range(5)
	
	for seed in seeds:

		### Load data
		np.random.seed(seed)
		X, L, labels, W_x, S_x, W_x_large, S_x_large = load_data(data_name = data_name, euc_th = euc_th, wl_th = wl_th, seed = seed)
		labeled_inds = np.random.choice(range(len(X)), size=num_labels, replace=False)

		W_wl, pseudolabs_soft = get_auggraph(L, labels, S_x_large, a, seed=seed)
		W_x_wl = W_x + mu * W_wl
		S_x_wl = normalize_matrix(W_x_wl)

		# compute smoothed aggregation
		smooth_pseudolabs_soft = smoothing_wl(pseudolabs_soft, labels, W_x_large)

		### Label propagation
		con_idx  = KhopNeighbor(labeled_inds, S_x, 1000) # get all connected
		con_idx2 = KhopNeighbor(labeled_inds, S_x_wl, 1000) # get all connected 
		alpha = 1/lamb

		# compute confidence score
		if step:
			prior_reg = compute_step_prior(smooth_pseudolabs_soft, labeled_inds, W_x, num_hops=3, step_val=0.8)
		else:
			prior_reg = compute_prior_reg(smooth_pseudolabs_soft, labeled_inds, W_x, num_hops=3)

		# WL
		base_preds_wlb = pseudolabs_soft.copy()
		base_preds_wlb[labeled_inds,:] = np.stack((1-labels[labeled_inds], labels[labeled_inds]), axis = 1)
		results.append(['wl', AdjustAcc(base_preds_wlb, labels), \
						NotAbstainAcc(base_preds_wlb, labels), \
						(base_preds_wlb[:,1]!= 0.5).sum() / labels.shape[0], 0, 0, "train"])
		
		if normalize:
			# LP
			f_baseline = PropagationAdaptive(np.ones_like(pseudolabs_soft)*0.5, S_x, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_baseline, labels, con_idx, "lp", "train"))
			# LP + WL
			f_lp_wl = PropagationAdaptive(pseudolabs_soft, S_x, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_wl, labels, con_idx, "lp+wl", "train"))
			# LPAG
			f_lp_ag = PropagationAdaptive(np.ones_like(pseudolabs_soft)*0.5, S_x_wl, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_ag, labels, con_idx2, "lp+ag", "train"))
			# LPAG + WL
			f_lp_ag_wl = PropagationAdaptive(pseudolabs_soft, S_x_wl, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_ag_wl, labels, con_idx2, "lp+ag+wl", "train"))
		else:
			# LP
			f_baseline = PropagationAdaptive(np.ones_like(pseudolabs_soft)*0.5, W_x, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_baseline, labels, con_idx, "lp", "train"))
			# LP + WL
			f_lp_wl = PropagationAdaptive(pseudolabs_soft, W_x, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_wl, labels, con_idx, "lp+wl", "train"))
			# LPAG
			f_lp_ag = PropagationAdaptive(np.ones_like(pseudolabs_soft)*0.5, W_x_wl, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_ag, labels, con_idx2, "lp+ag", "train"))
			# LPAG + WL
			f_lp_ag_wl = PropagationAdaptive(pseudolabs_soft, W_x_wl, labels, labeled_inds, alpha, prior_reg)
			results.append(get_results(f_lp_ag_wl, labels, con_idx2, "lp+ag+wl", "train"))
	
		if step:
			# saving data
			np.save("datasets/" + data_name +  "/step/" + "/baseline_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_baseline)
			np.save("datasets/" + data_name +  "/step/" + "/lpwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_wl)
			np.save("datasets/" + data_name +  "/step/" + "/lpag_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag)
			np.save("datasets/" + data_name +  "/step/" + "/lpagwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag_wl)
		else:
			if normalize:
				np.save("datasets/" + data_name +  "/baseline_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_baseline)
				np.save("datasets/" + data_name +  "/lpwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_wl)
				np.save("datasets/" + data_name +  "/lpag_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag)
				np.save("datasets/" + data_name +  "/lpagwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag_wl)
			else:
				np.save("datasets/" + data_name + "/no_norm" +  "/baseline_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_baseline)
				np.save("datasets/" + data_name +  "/no_norm" + "/lpwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_wl)
				np.save("datasets/" + data_name +  "/no_norm" + "/lpag_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
							"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag)
				np.save("datasets/" + data_name + "/no_norm" + "/lpagwl_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
						"_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(num_labels) + "_seed_" + str(seed), f_lp_ag_wl)
	return results

def exp_wrapper(data_name,  num_labels = 100, euc_th = 2, wl_th = 10, a = 0.2, mu = 0.01, lamb = 0.001):
	'''
	Wrapper to call and plot experiments
	'''	
	
	result = experiment(data_name = data_name,  num_labels = num_labels, euc_th = euc_th, wl_th = wl_th, a = a, mu = mu, lamb = lamb)
	df = pd.DataFrame(result, columns= ['Method','AdjustedAcc', 'NonAbstainAcc', 'Cov', 'ConnectedAcc', 'ConnectedCov', "split"])
	
	df_train = df[df.split == "train"]
	print({'data_name' : data_name,  'num_labels' : num_labels, 'euc_th' : euc_th, 'wl_th' : wl_th, 'a' : a, 'mu' : mu, 'lamb' : lamb})
	print("Train:")
	print(100*df_train.groupby('Method').mean())

	sns.barplot(data = df_train, x = 'Method', y ='AdjustedAcc' ,palette = "Set2")
	plt.ylim(0.5, 0.95)
	plt.title( "Train:" + data_name )
	# plt.show()

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default="youtube", type=str, help="Dataset to run (spam, agnews, yelp, awa2)")
	parser.add_argument('--num_labels', default=100, type=int, help="Number of initial labels coefficient")
	parser.add_argument('--step', action='store_true', help="Run with step prior")    
	parser.add_argument('--norm', action='store_true', help="Run with normalizing adjacency matrix")    
	args = parser.parse_args()

	# iterating and producing pseudolabels for all hyperparameter values
	for euc_th in [1, 10, 100]:
		for wl_th in [10, 100]:
			for a in [0.0, 0.5, 0.9]:
				for lamb in [0.001, 0.01]:
					if args.step:
						gen_pl(args.dataset,  num_labels = args.num_labels, euc_th = euc_th, wl_th = wl_th, a = a, mu = lamb, lamb = lamb, step=args.step)
						print("Generated", euc_th, wl_th, a, lamb)
					else:
						if args.norm:
							gen_pl(args.dataset,  num_labels = args.num_labels, euc_th = euc_th, wl_th = wl_th, a = a, mu = lamb, lamb = lamb, normalize=True)
						else:
							gen_pl(args.dataset,  num_labels = args.num_labels, euc_th = euc_th, wl_th = wl_th, a = a, mu = lamb, lamb = lamb, normalize=False)
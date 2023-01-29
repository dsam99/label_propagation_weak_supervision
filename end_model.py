import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

import models
import torch.optim
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import gc
import warnings 

import pickle

from snorkel.classification import cross_entropy_with_probs
from snorkel.labeling.model import LabelModel
from CLL.train_CLL import train_algorithm
from CLL.model_utilities import majority_vote_signal, set_up_constraint, get_error_bounds, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_end_model(train_data, pseudolabels, val_data, val_labels, test_data, test_labels, seed=0, soft=True):
	'''
	Function to train end model (MLP) with hyperparameter optimization

	Args:
	- train data, pseudolabels : training data and corresponding set of (soft) pseudolabels
	- val_data, val_labels : validation data with hard labels
	- test_data, test_labels : test data with hard labels
	(* data type expected is a torch tensor for each input)
	'''
	
	# constructing loaders
	train_loader = DataLoader(TensorDataset(train_data, pseudolabels), batch_size=100, shuffle=True)
	val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=100, shuffle=False)
	test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=100, shuffle=False)
	
	num_trials = 36
	val_accs, test_accs = np.zeros(num_trials), np.zeros(num_trials)
	loss_func = cross_entropy_with_probs
	hard_loss_func = nn.CrossEntropyLoss()

	def mlp_objective(trial):
		
		model = models.NeuralNetwork(train_data.shape[1], 2).to(device)

		num_epochs = trial.suggest_categorical("num_epochs", [20, 30, 40, 50])
		lr = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])

		reg_param = trial.suggest_categorical("reg param", [1e-2, 1e-3, 0])
		optimizer = torch.optim.Adam(model.parameters(), weight_decay=reg_param, lr=lr)

		# training NN
		for i in range(num_epochs): 
			model.train()
			for x, y in train_loader:
				optimizer.zero_grad()

				x, y = x.to(device), y.to(device)
				outputs = model(x)
				
				if soft:
					loss = loss_func(outputs, y)
				else:
					loss = hard_loss_func(outputs, y)

				loss.backward()
				optimizer.step()

		# Validation of the model
		model.eval()
		val_preds = []
		test_preds = []
		with torch.no_grad():
			for x, _ in val_loader:
				x = x.to(device)
				val_pred = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
				val_preds.append(val_pred)

			for x, _ in test_loader:
				x = x.to(device)
				test_pred = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
				test_preds.append(test_pred)

		val_preds = np.concatenate(val_preds)
		val_accuracy = np.mean(val_preds == val_labels.numpy())
		test_preds = np.concatenate(test_preds)
		test_accuracy = np.mean(test_preds == test_labels.numpy())

		# storing val & test accuracy for later
		val_accs[trial.number] = val_accuracy
		test_accs[trial.number] = test_accuracy
		return val_accuracy

	sampler = TPESampler(seed = seed) 
	nn_study = optuna.create_study(sampler=sampler, direction="maximize")
	nn_study.optimize(mlp_objective, n_trials=num_trials, timeout=50000, gc_after_trial=True)

	best_ind = nn_study.best_trial.number
	print("Trial", best_ind, "Best Val Acc:", val_accs[best_ind], "Test Acc", test_accs[best_ind])	
	return val_accs[best_ind], test_accs[best_ind]

def run_cll_model(train_data, train_L, train_labels, labeled_inds, val_data, val_labels, test_data, test_labels, seed):

	'''
	Function to train end model (MLP) with hyperparameter optimization on labels from CLL

	Args:
	- train data, pseudolabels : training data and corresponding set of (soft) pseudolabels
	- val_data, val_labels : validation data with hard labels
	- test_data, test_labels : test data with hard labels
	(* data type expected is a torch tensor for each input)
	'''

	m, n, k = train_L.shape
	weak_errors = np.ones((m, k)) * 0.01
	
	# get constraints via small labeled data
	weak_errors = get_error_bounds(train_labels[labeled_inds], train_L[:, labeled_inds, :])
	weak_errors = np.asarray(weak_errors)

	# Set up the constraints
	constraints = set_up_constraint(train_L, weak_errors)
	constraints['weak_signals'] = train_L

	y = train_algorithm(constraints)
	accuracy = accuracy_score(train_labels, y)
	print("Constrained Label Accs:", accuracy)

	soft = True
	constrained_pls = np.stack([1 - y.flatten(), y.flatten()], axis=1)
	num_trials = 36 
	val_accs, test_accs = np.zeros(num_trials), np.zeros(num_trials)

	train_loader = DataLoader(TensorDataset(train_data, torch.tensor(constrained_pls, dtype=torch.float)), batch_size=100, shuffle=True)
	val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=100, shuffle=False)
	test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=100, shuffle=False)
	loss_func = cross_entropy_with_probs

	def mlp_objective(trial):
		
		model = models.NeuralNetwork(train_data.shape[1], 2).to(device)

		num_epochs = trial.suggest_categorical("num_epochs", [20, 30, 40, 50])
		lr = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
		reg_param = trial.suggest_categorical("reg param", [1e-2, 1e-3, 0])
		optimizer = torch.optim.Adam(model.parameters(), weight_decay=reg_param, lr=lr)

		# training NN
		for i in range(num_epochs): 
			model.train()
			for x, y in train_loader:
				x, y = x.to(device), y.to(device)
				optimizer.zero_grad()

				outputs = model(x)
				if soft:
					loss = loss_func(outputs, y)
				else:
					loss = hard_loss_func(outputs, y)
				loss.backward()
				optimizer.step()

		# Validation of the model
		model.eval()
		val_preds = []
		test_preds = []
		with torch.no_grad():
			for x, _ in val_loader:
				x = x.to(device)
				val_pred = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
				val_preds.append(val_pred)

			for x, _ in test_loader:
				x = x.to(device)
				test_pred = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
				test_preds.append(test_pred)

		val_preds = np.concatenate(val_preds)
		val_accuracy = np.mean(val_preds == val_labels.numpy())
		test_preds = np.concatenate(test_preds)
		test_accuracy = np.mean(test_preds == test_labels.numpy())

		# storing val & test accuracy for later
		val_accs[trial.number] = val_accuracy
		test_accs[trial.number] = test_accuracy
		return val_accuracy

	sampler = TPESampler(seed = seed) 
	nn_study = optuna.create_study(sampler=sampler, direction="maximize")
	nn_study.optimize(mlp_objective, n_trials=num_trials, timeout=50000, gc_after_trial=True)

	best_ind = nn_study.best_trial.number
	print("Trial", best_ind, "Best Val Acc:", val_accs[best_ind], "Test Acc", test_accs[best_ind])
	return val_accs[best_ind], test_accs[best_ind]


if __name__ == "__main__":

	import dataset
	import argparse
	parser = argparse.ArgumentParser()	
	parser.add_argument('--dataset', default="youtube", type=str, help="Dataset to run (spam, agnews, yelp, awa2)")
	parser.add_argument('--method', default="lpag", type=str, help="Which method to load pseudolabels from")
	parser.add_argument('--euc_th', default=2, type=int, help="Euclidean distance threshold")
	parser.add_argument('--wl_th', default=20, type=int, help="Weak Labeler partition threshold")
	parser.add_argument('--a', default=0.0, type=float, help="Abstain threshold")
	parser.add_argument('--mu', default=0.001, type=float, help="Weighting of weak labeler graph")
	parser.add_argument('--lamb', default=0.001, type=float, help="Prior regularization coefficient")
	parser.add_argument('--num_labels', default=100, type=int, help="Number of initial labels coefficient")    
	parser.add_argument('--step', action='store_true', help="Run with step prior")    
	parser.add_argument('--norm', action='store_true', help="Run with normalized adj matrix")    
	args = parser.parse_args()

	seeds = [0, 1, 2, 3, 4]
	res = []
	soft = True

	# loop over euc_th and wl_th
	for euc_th in [1, 10, 100]:
		for wl_th in [10, 100]:
			for method in ["baseline", "lpag", "lpwl", "lpagwl"]:
				
				print("Running for:", euc_th, wl_th, method)
				
				for seed in seeds:

					# setting seeds
					torch.manual_seed(seed)
					torch.cuda.manual_seed(seed)
					torch.backends.cudnn.deterministic = True
					torch.backends.cudnn.benchmark = False
					np.random.seed(seed)

					train_data = torch.load("datasets/" + args.dataset + "/train_X_seed" + str(seed))
					train_labels = torch.load("datasets/" + args.dataset + "/train_labels_seed" + str(seed)).numpy()

					if method == "sup":
						pseudolabs = train_labels
					else:
						if not args.step:
							pl_path = "datasets/" + args.dataset + "/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(args.a) + \
									"_mu_" + str(args.mu) + "_lambda_" + str(args.lamb) + "_numlab_" + str(args.num_labels) + "_seed_" + str(seed) + ".npy"
						else:
							if args.norm:
								pl_path = "datasets/" + args.dataset + "/step/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(args.a) + \
									"_mu_" + str(args.mu) + "_lambda_" + str(args.lamb) + "_numlab_" + str(args.num_labels) + "_seed_" + str(seed) + ".npy"
							else:
								pl_path = "datasets/" + args.dataset + "/no_norm/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(args.a) + \
									"_mu_" + str(args.mu) + "_lambda_" + str(args.lamb) + "_numlab_" + str(args.num_labels) + "_seed_" + str(seed) + ".npy"
							# load training data    
						pseudolabs = np.load(pl_path)
						# filtering out points pseudolabels abstain on
						valid_inds = np.abs(pseudolabs[:, 0] - 0.5) > 0.001			
						# print("Coverage", np.sum(valid_inds) / len(pseudolabs))
						
						train_data = train_data[valid_inds]
						pseudolabs = pseudolabs[valid_inds]
						train_labels = train_labels[valid_inds]
						
						hard_pseudolabs = np.argmax(pseudolabs, axis=1)
						# print("PL Accuracy", np.mean(hard_pseudolabs == train_labels))

					if args.dataset == "basketball" or args.dataset == "tennis":
						# _, val_dataset, test_dataset = get_dataset(args.dataset, feature=None)
						val_dataset = dataset.WSDataset(dataset=args.dataset, split="val", feature=None, balance=False, seed = seed)
						test_dataset = dataset.WSDataset(dataset=args.dataset, split="test", feature=None, balance=False, seed = seed)
					else:
						# _, val_dataset, test_dataset = get_dataset(args.dataset, feature="bert")
						val_dataset = dataset.WSDataset(dataset=args.dataset, split="val", feature="bert", balance=False, seed = seed)
						test_dataset = dataset.WSDataset(dataset=args.dataset, split="test", feature="bert", balance=False, seed = seed)

					# getting data / labels seperately
					val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
					val_data, _, val_labels = next(iter(val_loader))
					test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
					test_data, _, test_labels = next(iter(test_loader))

					# print(train_data.shape, val_data.shape, test_data.shape)
					# print(pseudolabs)
					# running end model with soft labels
					if args.method == "sup":
						best_val, best_test = run_end_model(train_data, torch.tensor(train_labels, dtype=torch.long), val_data, val_labels, test_data, test_labels, soft=False)
					else:
						if soft:
							best_val, best_test = run_end_model(train_data, torch.tensor(pseudolabs, dtype=torch.float), val_data, val_labels, test_data, test_labels, soft=True)
						else:
							best_val, best_test = run_end_model(train_data, torch.tensor(hard_pseudolabs, dtype=torch.long), val_data, val_labels, test_data, test_labels, soft=False)
					
					# print("Val", best_val, "Test", best_test)
					res.append(best_test)
					# running end model with hard labels
					
					path = "results/" + args.dataset + "_results"
					if args.step:
						path += "_step"

					if not args.norm:
						path += "_nonorm"

					# if method == "sup":
					# 	pass
					# else:
					# 	if soft:
					# 		path += "_soft.txt"
					# 	else:
					# 		path += "_hard.txt"
					# 	with open(path, "a") as f:
					# 		res = [args.dataset, method, seed, best_val, best_test, euc_th, wl_th, args.a, args.mu, args.lamb]
					# 		# print("".join([str(x) for x in res], ))
					# 		f.write(",".join([str(x) for x in res]))
					# 		f.write("\n")
				for i in res:
					print(i)
				
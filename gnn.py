
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn import GraphConv
from main import load_data
from torch.nn import Softmax
import dataset
import sys

from end_model import run_end_model

class GCN(nn.Module):
	def __init__(self, in_feats, h_feats, num_classes):
		super(GCN, self).__init__()
		self.conv1 = GraphConv(in_feats, h_feats)
		self.conv2 = GraphConv(h_feats, num_classes)
	
	def forward(self, g, in_feat):
		h = self.conv1(g, in_feat)
		h = F.relu(h)
		h = self.conv2(g, h)
		return h

def train(g, model):

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
	best_val_acc = 0
	best_test_acc = 0

	features = g.ndata['feat']
	labels = g.ndata['label']
	train_mask = g.ndata['train_mask']
	val_mask = g.ndata['val_mask']
	test_mask = g.ndata['test_mask']


	for e in range(200):
		# Forward
		logits = model(g, features)

		# Compute prediction
		pred = logits.argmax(1)

		# Compute loss
		# Only compute losses of the nodes
		loss = F.cross_entropy(logits[train_mask], labels[train_mask])

		# Compute accuracy on training/ val / test
		train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
		val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
		test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

		# Save the best validation accuracy and the corresponding test accuracy
		if best_val_acc < val_acc:
			best_val_acc = val_acc
			best_test_acc = test_acc
		
		# Backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# if e % 20 == 0:
		#     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.\
	#     #         format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
	# print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.\
	#             format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

	model.eval()
	final_pred = model(g, features)

	return test_acc, final_pred

def eval(g):
	model = GCN(g.ndata['feat'].shape[1], 16, 2)
	test_acc, final_pred = train(g, model)
	return test_acc, final_pred


def adjmat_to_graph_wl(W_x, X, L, labels, labeled_inds,i):
	labeled_inds_array = torch.tensor([i in labeled_inds for i in range(len(X))])
	num_nodes = W_x.shape[0]
	src, dst = np.nonzero(W_x)
	g = dgl.graph((src, dst), num_nodes = num_nodes)
	g.ndata['feat'] = X
	g.ndata['label'] = labels

	idx = (L[:,i]!= -1)  

	g.ndata['train_mask'] = labeled_inds_array*idx
	unlabeled_inds_array = ~labeled_inds_array
	g.ndata['val_mask'] = unlabeled_inds_array*idx
	g.ndata['test_mask'] = unlabeled_inds_array*idx

	# add a self-loop for stability
	g = dgl.add_self_loop(g)

	return g    


def adjmat_to_graph(W_x, X, labels, labeled_inds):
	labeled_inds_array = torch.tensor([i in labeled_inds for i in range(len(X))])
	num_nodes = W_x.shape[0]
	src, dst = np.nonzero(W_x)
	g = dgl.graph((src, dst), num_nodes = num_nodes)
	g.ndata['feat'] = X
	g.ndata['label'] = labels

	g.ndata['train_mask'] = labeled_inds_array
	unlabeled_inds_array = ~labeled_inds_array
	g.ndata['val_mask'] = unlabeled_inds_array
	g.ndata['test_mask'] = torch.tensor(np.array([True for _ in range(num_nodes)]))

	# add a self-loop for stability
	g = dgl.add_self_loop(g)

	return g


if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()	
	parser.add_argument('--dataset', default="youtube", type=str, help="Dataset to run (spam, agnews, yelp, awa2)")
	parser.add_argument('--soft', action="store_true", help="Run with soft pseudolabels")
	args = parser.parse_args()

	args.pretrained = True

	print("Dataset:", args.dataset)
	print("Soft Labels:", args.soft)

	num_labels = 100 # fixed size

	seeds = list(range(5))

	for seed in seeds[2:3]:

		euc_ths = [1, 10, 100]

		val_accs = []
		test_accs = []

		for euc_th in euc_ths:

			# setting seeds
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			np.random.seed(seed)

			X, L, labels, W_x, _,_,_ = load_data(data_name = args.dataset, euc_th = euc_th, wl_th = 10, seed = seed)

			labeled_inds = np.random.choice(range(len(X)), size=num_labels, replace=False)
			g = adjmat_to_graph(W_x, X, labels, labeled_inds)
			acc, final_pred = eval(g)
			softmax = Softmax()

			with torch.no_grad():
				pseudo_labels = softmax(final_pred)[:,1]

			# detaching for re-training
			X = X.detach()
			pseudo_labels = pseudo_labels.detach()

			# print('Acc :', ((pseudo_labels > 0.5001).long() == labels).float().mean())

			# loading val and test data
			if args.dataset == "basketball" or args.dataset == "tennis":
				val_dataset = dataset.WSDataset(dataset=args.dataset, split="val", feature=None, balance=False, seed = seed)
				test_dataset = dataset.WSDataset(dataset=args.dataset, split="test", feature=None, balance=False, seed = seed)
			else:
				val_dataset = dataset.WSDataset(dataset=args.dataset, split="val", feature="bert", balance=False, seed = seed)
				test_dataset = dataset.WSDataset(dataset=args.dataset, split="test", feature="bert", balance=False, seed = seed)

			# getting data / labels seperately
			val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
			val_data, _, val_labels = next(iter(val_loader))
			test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
			test_data, _, test_labels = next(iter(test_loader))

			hard_pl = torch.argmax(softmax(final_pred), dim=1)

			if args.soft:
				best_val, best_test = run_end_model(X, softmax(final_pred).detach(), val_data, val_labels, test_data, test_labels, soft=True)
			else:
				best_val, best_test = run_end_model(X, hard_pl, val_data, val_labels, test_data, test_labels, soft=False)
			
			val_accs.append(best_val)
			test_accs.append(best_test)

		print("Seed", seed, "Val", val_accs, "Test", test_accs)



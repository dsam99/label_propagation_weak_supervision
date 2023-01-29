import numpy as np
import sys
import torch

sys.path.append("./wrench")
from wrench.dataset import load_dataset
from imblearn.under_sampling import RandomUnderSampler

def get_dataset(data_name, feature="bert"):
	'''
	Function to load the dataset given by data_name

	Args:
	data_name - string representing dataset
	feature - if should be loaded with bert or No extractor
	'''

	#### Load dataset 
	dataset_home = './datasets'

	if feature == "bert":
		extract_fn = 'bert'
		model_name = 'bert-base-cased'
		
		train_data, valid_data, test_data = load_dataset(dataset_home, data_name, extract_feature=True, extract_fn=extract_fn,
													cache_name=extract_fn, model_name=model_name)
	
	else:
		train_data, valid_data, test_data = load_dataset(dataset_home, data_name, extract_feature=True, extract_fn=None)
	
	return train_data, valid_data, test_data

class WSDataset(torch.utils.data.Dataset):
	'''
	Class for a (balanced) Weakly Supervised Datasets
	'''
	
	def __init__(self, dataset="youtube", split='train', feature="bert", image = False, balance=False, seed = 0):


		train, val, test = get_dataset(dataset, feature)

		if image:
			train_data = np.array(train.examples)
			val_data = np.array(val.examples) 
			test_data = np.array(test.examples)

		else:
			train_data = np.array(train.features)
			val_data = np.array(val.features) 
			test_data = np.array(test.features)
		
		train_votes, train_labels = np.array(train.weak_labels), np.array(train.labels)
		val_votes, val_labels = np.array(val.weak_labels), np.array(val.labels)
		test_votes, test_labels = np.array(test.weak_labels), np.array(test.labels)

		if balance:
			rus = RandomUnderSampler(random_state = seed)
			train_ind, train_labels = rus.fit_resample(np.reshape(list(range(len(train_data))), (-1, 1)), train_labels)
			val_ind, val_labels = rus.fit_resample(np.reshape(list(range(len(val_data))), (-1, 1)), val_labels)
			test_ind, test_labels = rus.fit_resample(np.reshape(list(range(len(test_data))), (-1, 1)), test_labels)

			train_data, train_votes = train_data[train_ind.flatten()], train_votes[train_ind.flatten()]
			val_data, val_votes = val_data[val_ind.flatten()], val_votes[val_ind.flatten()]
			test_data, test_votes = test_data[test_ind.flatten()], test_votes[test_ind.flatten()]
		
		if split == "train":
			self.labels = torch.tensor(train_labels, dtype=torch.long)
			self.X = torch.tensor(train_data, dtype=torch.float)
			self.L = torch.tensor(train_votes, dtype=torch.long)

		elif split == "val":
			self.labels = torch.tensor(val_labels, dtype=torch.long)
			self.X = torch.tensor(val_data, dtype=torch.float)
			self.L = torch.tensor(val_votes, dtype=torch.long)
		
		elif split == "test":
			self.labels = torch.tensor(test_labels, dtype=torch.long)
			self.X = torch.tensor(test_data, dtype=torch.float)
			self.L = torch.tensor(test_votes, dtype=torch.long)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		X = self.X[index, :]
		L = self.L[index, :]
		label = self.labels[index]
		
		return X, L, label

if __name__ == "__main__":
	youtube_data = WSDataset("youtube", "train")
	print(youtube_data[0])
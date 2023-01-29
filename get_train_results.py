#### Python Script to take a CSV of end_model results and print pseudolabel train results 

import pandas as pd
import numpy as np
import torch
from lpa.utils import AdjustAcc, NotAbstainAcc
from snorkel.labeling.model import LabelModel

methods = ["baseline", "lpag", "lpwl", "lpagwl"]

wl = False
if wl:

    for dset in ["youtube", "sms", "cdr", "basketball", "tennis"]:
        
        base_path = "datasets/" + dset + "/"
        print("Dataset", dset)

        # looping through seeds
        accs = []
        abs_accs = []
        covs = []
        print("Base WL")
        for i in range(5):
            train_L = torch.load(base_path + "/train_L_seed" + str(i)).numpy()
            train_labels = torch.load(base_path + "/train_labels_seed" + str(i)).numpy()

            # getting wl predictions
            label_model = LabelModel(cardinality=2, verbose=False)
            label_model.fit(train_L, n_epochs=200, seed=i)
            pseudolabs = label_model.predict_proba(train_L)

            accs.append(AdjustAcc(pseudolabs, train_labels))
            abs_accs.append(NotAbstainAcc(pseudolabs, train_labels))
            covs.append(np.sum(np.abs(pseudolabs[:,1] - 0.5) > 0.001) / train_labels.shape[0])

        print("Acc")
        for i in accs:
            print(i)
        print("NonAbs Acc")
        for i in abs_accs:
            print(i)
        print("Cov")
        for i in covs:
            print(i)

else:
    for dset in ["youtube", "sms", "cdr", "basketball", "tennis"]:
        
        base_path = "datasets/" + dset + "/"
        print("Dataset", dset)

        path = "Hyperparams_LP Experiments - " + dset + "_nonorm.csv"
        df = pd.read_csv(path)

        res = {m:[] for m in methods}
        for i, row in df.iterrows():
            # if i != 0:
            m = row[2]
            hyp = (int(row[6]), int(row[7]), float(row[8]), float(row[9]))
            res[m].append(hyp)
        
        # print(res)
        for method in methods:
            
            print("Method", method)
            print("Adjust Acc")
            for i, r in enumerate(res[method]):

                train_labels = torch.load(base_path + "/train_labels_seed" + str(i)).numpy()
                euc_th, wl_th, a, mu = r
                lamb = mu
                pl_path = "datasets/" + dset + "/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
                    "_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(100) + "_seed_" + str(i) + ".npy"
                pseudolabs = np.load(pl_path)
                print(AdjustAcc(pseudolabs, train_labels))
                
            print("Not Abstain Acc")
            for i, r in enumerate(res[method]):

                train_labels = torch.load(base_path + "/train_labels_seed" + str(i)).numpy()
                euc_th, wl_th, a, mu = r
                lamb = mu
                pl_path = "datasets/" + dset + "/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
                    "_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(100) + "_seed_" + str(i) + ".npy"
                pseudolabs = np.load(pl_path)
                print(NotAbstainAcc(pseudolabs, train_labels))
            
            print("Cov")
            for i, r in enumerate(res[method]):
                train_labels = torch.load(base_path + "/train_labels_seed" + str(i)).numpy()
                euc_th, wl_th, a, mu = r
                lamb = mu
                pl_path = "datasets/" + dset + "/" + method + "_euc_" + str(euc_th) + "_wl_" + str(wl_th) + "_a_" + str(a) + \
                    "_mu_" + str(mu) + "_lambda_" + str(lamb) + "_numlab_" + str(100) + "_seed_" + str(i) + ".npy"
                pseudolabs = np.load(pl_path)   
                print(np.sum(np.abs(pseudolabs[:,1] - 0.5) > 0.001) / train_labels.shape[0])

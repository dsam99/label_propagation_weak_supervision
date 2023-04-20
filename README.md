# label_propagation_weak_supervision
Code repository for the paper [Label Propagation with Weak Supervision](https://arxiv.org/abs/2210.03594) (ICLR 2023)!

## Data 

To replicate our experiments, you first need to download the data from the WRENCH benchmark (youtube, sms, cdr, basketball, tennis). These can be found at this link: 
https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq

## Running our Code

To run our experiments, you can use the command

```
python gen_pseudolabels.py --dataset youtube
```

This will run our algorithm (and other baselines) and generate sets of pseudolabels for the data points. This will also save copies of the underlying graphs required for our method and the standard LPA algorithm. We note that this can be time-consuming and memory demanding, so a heads up if you are running this for the larger datasets (cdr, basektball, tennis). 

Then, you can train a neural network with these pseudolabels by running the following command:

```
python end_model.py --dataset youtube
```

## Citation

Please cite the following paper if you use our work. Thanks!

Rattana Pukdee*, Dylan Sam*, Maria-Florina Balcan, and Pradeep Ravikumar. Label Propagation with Weak Supervision. <em>International Conference on Learning Representations (ICLR)</em>, 2023.

```
@article{pukdee2022label,
  title={Label Propagation with Weak Supervision},
  author={Pukdee, Rattana and Sam, Dylan and Balcan, Maria-Florina and Ravikumar, Pradeep},
  journal={arXiv preprint arXiv:2210.03594},
  year={2022}
}
```

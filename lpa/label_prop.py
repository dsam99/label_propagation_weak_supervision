import numpy as np
from utils import KhopNeighbor

def smoothing_wl(base_preds, labels, W):
    # Weak labels aggregation
    smooth_wl = PropagationSoft(base_preds, W , labels, labeled_inds = [], alpha = 10)
    return smooth_wl

def compute_prior_reg(smooth_wls, labeled_inds, W, num_hops=1):
    '''
    Compute prior regularization value

    Args:
    smooth_wls - smoothed weak labeler aggregation 
    W - graph 
    '''

    khop_inds = KhopNeighbor(labeled_inds, W, num_hops)
    score = smooth_wls[:,1]
    score = 2 * np.abs(score - 0.5) + 0.001
    unconfidence_lambda = 0.001
    score[khop_inds] *= 0.001
    # return np.exp((score - 1))
    return np.square(score)

def compute_step_prior(smooth_wls, labeled_inds, W, num_hops=3, step_val=0.2):
    '''
    Compute prior regularization value - via a step function

    Args:
    smooth_wls - smoothed weak labeler aggregation 
    W - graph 
    '''

    khop_inds = KhopNeighbor(labeled_inds, W, num_hops)
    
    score = smooth_wls[:,1]
    prior = np.ones_like(score) * 0.001
    prior[score < step_val] = 1e6
    prior[score > 1 - step_val] = 1e6
    prior[khop_inds] *= 0.001 # resetting points that are within k-hops from labeled points
    return prior

##### Label Propagation #####
def PropagationSoft(base_preds, W, labels, labeled_inds, alpha):

    '''
    Function to Perform LPA with soft constraints

    Args:
    base_preds - initialization (i.e, prior from weak labelers)
    W - graph
    labels - true labels (only use a small amount)
    labeled_inds - indices corresponding to true labels that we use
    alpha - amount of weighting used for the LPA
    '''

    # Propagation with soft constraint
    unlabeled_inds = np.array(list(set(range(base_preds.shape[0])) - set(labeled_inds)))
    f_l = np.stack((1 - labels[labeled_inds], labels[labeled_inds]), axis=1).astype(float)
    f_u = base_preds[unlabeled_inds,:]

    f_0 = np.zeros((labels.shape[0], 2))
    f_0[labeled_inds,: ] = f_l
    f_0[unlabeled_inds, :] = f_u

    D = W.sum(axis = 1)*np.identity(W.shape[0])
    S_ = (W-D)
    f_smooth = np.matmul(np.linalg.inv(np.identity(D.shape[0]) - alpha*S_), f_0)
    # f_smooth = IterativeMethod( f_0 = f_0, S = S_, alpha = alpha, num_iter = 100)
    return f_smooth

def PropagationHard(base_preds, W, labels, labeled_inds , alpha):
    '''
    Function to Perform LPA with hard constraints

    Args:
    base_preds - initialization (i.e, prior from weak labelers)
    W - graph
    labels - true labels (only use a small amount)
    labeled_inds - indices corresponding to true labels that we use
    alpha - amount of weighting used for the LPA
    '''    

    # Propagation with hard constraint
    unlabeled_inds = np.array(list(set(range((base_preds).shape[0])) - set(labeled_inds)))
    f_l = np.stack((1 - labels[labeled_inds], labels[labeled_inds]), axis=1).astype(float)
    f_u_0 = base_preds[unlabeled_inds,:]

    W_uu = W[unlabeled_inds][:,unlabeled_inds]
    W_lu = W[labeled_inds][:,unlabeled_inds]
    W_ul = W[unlabeled_inds][:,labeled_inds]

    D_ul = W_ul.sum(axis = 1)* np.identity(W_ul.shape[0])
    D_uu = W_uu.sum(axis = 1)* np.identity(W_uu.shape[0])

    S_ = (W_uu - D_uu - D_ul)
    f_u = np.matmul(np.linalg.inv(np.identity(D_uu.shape[0]) - alpha*S_) , f_u_0+  alpha*np.matmul(np.transpose(W_lu), f_l))

    f_smooth = np.zeros((labels.shape[0], 2))
    f_smooth[labeled_inds,: ] = f_l
    f_smooth[unlabeled_inds, :] = f_u

    return f_smooth

def PropagationAdaptive(base_preds, W, labels, labeled_inds, alpha, confidence_score):
    # Propagation with hard constraint and soft constraint with coefficient depending on confidence score lambda
    unlabeled_inds = np.array(list(set(range((base_preds).shape[0])) - set(labeled_inds)))
    f_l = np.stack((1 - labels[labeled_inds], labels[labeled_inds]), axis=1).astype(float)
    f_u_0 = base_preds[unlabeled_inds,:]

    W_uu = W[unlabeled_inds][:,unlabeled_inds]
    W_lu = W[labeled_inds][:,unlabeled_inds]
    W_ul = W[unlabeled_inds][:,labeled_inds]

    D_ul = W_ul.sum(axis = 1)* np.identity(W_ul.shape[0])
    D_uu = W_uu.sum(axis = 1)* np.identity(W_uu.shape[0])

    # confidence score
    D_lambda = alpha * confidence_score[unlabeled_inds]*np.identity(len(unlabeled_inds))

    S_ = (D_uu - W_uu + D_ul + 0.5*D_lambda)
    f_u = np.matmul(np.linalg.inv(S_), np.matmul(np.transpose(W_lu), f_l) + 0.5*np.matmul(D_lambda, f_u_0))

    f_smooth = np.zeros((labels.shape[0], 2))
    f_smooth[labeled_inds,: ] = f_l
    f_smooth[unlabeled_inds, :] = f_u

    return f_smooth

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances

##### Accuracy #####
def NotAbstainAcc(pred, labels):
    '''
    Function that computes accuracy on non-abstained regions 
    '''    
    valid_inds = np.abs((pred[:,0] - 0.5)) > 0.001
    acc = Acc(pred[valid_inds], labels[valid_inds])
    return acc
    
def AdjustAcc(pred, labels):
    '''
    Function that computes accuracy on all regions (i.e., setting acc on abstained regions to 50%)
    '''
    
    acc = NotAbstainAcc(pred, labels)
    valid_inds = np.abs((pred[:,0] - 0.5)) > 0.01
    n = np.sum(valid_inds)
    N = labels.shape[0]
    return acc * n / N + 0.5 * (N - n) / N

def Acc(pred, labels):
    '''
    Function that computes given predictions and true labels
    '''    
    return np.mean(np.argmax(pred, axis = 1) == np.array(labels))

##### Euclidean graph construction #####
def get_transition_mat(combined_dis_mat, threshold):
    '''
    Function that generates a adjacency matrix given a distance matrix and a particular threshold
    '''

    # Radius Neighbor 
    radius_mat = radius_neighbors_graph(combined_dis_mat, threshold, metric='precomputed', include_self=True)
    radius_mat = radius_mat.toarray()
    return radius_mat

def get_transition_mat_nn(combined_dis_mat, n_neighbors = 10):
    '''
    Function that generates a adjacency matrix given a distance matrix and a number of nearest neighbors (not used)
    '''

    # Nearest Neighbor
    neigh = NearestNeighbors(n_neighbors=5, metric="precomputed")
    neigh.fit(combined_dis_mat)
    NN_mat = neigh.kneighbors_graph(combined_dis_mat)
    NN_mat = NN_mat.toarray()
    return NN_mat

def GenerateMatrix(euc_mat, thresh = 10):
    '''
    Function to genereate an adjacency and a normalized adjacency matrix
    '''

    np.seterr(invalid='ignore')
    comb_mat = euc_mat
    N = euc_mat.shape[0]
    threshold = np.quantile(comb_mat, thresh/N)
    adj_mat= get_transition_mat(comb_mat, threshold)

    np.seterr(invalid='ignore')
    D_inv = 1/adj_mat.sum(axis = 1) * np.identity(adj_mat.shape[0])
    D_inv = np.nan_to_num(D_inv)
    # doing S = D^{-1/2} A D^{-1/2}
    S = np.matmul(np.matmul(np.sqrt(D_inv), adj_mat), np.sqrt(D_inv))
    S = np.nan_to_num(S)

    return adj_mat, S

def normalize_matrix(adj_mat):
    '''
    Helper function to normalized adjacency matrix
    '''    
    
    # full transition matrix
    # adj_mat = comb_mat
    np.seterr(invalid='ignore')
    D_inv = 1/adj_mat.sum(axis = 1) * np.identity(adj_mat.shape[0])
    D_inv = np.nan_to_num(D_inv)
    # doing S = D^{-1/2} A D^{-1/2}
    S = np.matmul(np.matmul(np.sqrt(D_inv), adj_mat), np.sqrt(D_inv))
    S = np.nan_to_num(S)
    return S


##### Neighbors #####

def GetNeighbor(A, S):

    '''
    Function to get the neighbors of a set A given graph S
    '''

    # look at the k hop neighbor of label points
    Neighbor_A = set()
    for v in A:
        Neighbor_A.update(np.where(S[v] != 0)[0])
    Neighbor_A = set(A).union(Neighbor_A)
    return np.array(list(Neighbor_A))

def KhopNeighbor(A,S, k):
    '''
    (Efficient version) Function to get the neighbors of a set A given graph S
    '''

    # more efficient code
    nb_list = [A, GetNeighbor(A,S)]
    for _ in range(k-1):
        new_points = list(set(nb_list[-1]) - set(nb_list[-2]))
        new_nb = GetNeighbor(new_points,S)
        total_nb = list(set(list(nb_list[-1]) + list(new_nb)))
        nb_list.append(total_nb)

        if len(nb_list[-1]) == len(nb_list[-2]):
            break
    return np.array(nb_list[-1])


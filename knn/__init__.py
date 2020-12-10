"""
API for k-nearest neighbors
"""

import numpy as np
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

class KNN:
    
    def __init__(self,X,y,labels,k):
        self.X = X
        self.y = y
        self.labels = labels
        self.k = k
        
    def euclidean_distance(self,X,x_0):
        """
        vectorized Euclidean distance
        
        1. reshape y to row vector so that it broadcasts downward for subtraction on each row
        2. square each subtraction and sum across rows --> column vector
        3. square root of column vector --> distances
        """
        return np.sum((X-x_0.reshape(1,-1))**2,axis=1)**0.5
    
    def majority_vote(self,dists):
        """
        1. pair dists and associated class in a dataframe
        2. sort the dataframe by distance in ascending order
        3. select the top k nearest neighbors
        4. majority vote is the mode of the knns
        
        returns votes in a list
        
        edge cases:
        equidistant to an equal number of points in multiple classes --> ambiguous vote
        """
        pdf = pd.DataFrame({'dists':dists,'y':self.y})
        pdf.sort_values(by=['dists'],inplace=True) # uses quicksort
        top_y = pdf.y[:self.k]
        mode = top_y.mode() # if len(mode) > 1, then there are ties
        
        ties = list(mode.values.astype(int))
        
        return random.choice(ties) # random choice of all ties
        
    
    def predict(self,x_0,dist='euclidean',vote='majority'):

        dists = None
        if dist == 'euclidean':
            dists = self.euclidean_distance(self.X,x_0)
        else:
            raise ValueError("Specified distance metric is undefined\n")
            
        cls = None
        if vote == 'majority':
            cls = self.majority_vote(dists)
        else:
            raise ValueError("Specified voting method is undefined\n")
        
        return cls
    
    def evaluate(self,Xt,yt):
        
        m = len(yt)
        correct = 0
        incorrect = []
        for i in range(m):
            cls = self.predict(Xt[i,:])
            if cls == yt[i]:
                correct += 1
            else:
                incorrect += [(i,self.labels[cls],self.labels[yt[i]])]
            
        
        return correct / m, incorrect

#################################################################################

# choose best K neighbors using K-Fold CV
def knn_cv(X,y,labels,seed,stratify,n_splits,k):
    
    if stratify:
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    else:
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    
    perfs= []
    for tr_ind, te_ind in kf.split(X,y):
        X_tr, X_te = X[tr_ind], X[te_ind]
        y_tr, y_te = y[tr_ind], y[te_ind]
        
        exp = KNN(X_tr,y_tr,labels,k)
        perf,_ = exp.evaluate(X_te,y_te)
        
        perfs += [perf]
    
    return sum(perfs) / n_splits

def choose_k(X,y,labels,seed,stratify,n_splits,leeway):
    
    max_k = len(X)
    
    k=1
    acc = 0
    cont = True
    best_k = k
    countdown = 0
    while cont and k <= max_k:
        perf = knn_cv(X,y,labels,seed,stratify,n_splits,k)
        
        if perf <= acc:
            cont = False if countdown > leeway else cont
            countdown += 1
        else:
            acc = perf
            best_k = k
            countdown = 0
        
        k += 1
    
    return best_k

#################################################################################

# demonstrative testing function
def test_knn(X,y,labels,seed,stratify=False,n_splits=10,leeway=5):
    
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.1,random_state=seed)
    print("Seed =",seed)
    
    t1 = time.time()
    K = choose_k(X_tr,y_tr,labels,seed,stratify,n_splits,leeway)
    knn = KNN(X_tr,y_tr,labels,K)
    test_perf, incorrect = knn.evaluate(X_te,y_te)
    debug_perf, debug_incorrect = knn.evaluate(X_tr,y_tr)
    t2 = time.time()
    
    print("Chosen k: ", K)
    print("Test accuracy: ",test_perf)
    print("Misclassified indices in the form (index, prediction, truth): ", incorrect)
    print("Debug accuracy: ", debug_perf)
    print("Misclassified indices in the form (index, prediction, truth): ", debug_incorrect)
    print("Time taken: ", t2-t1)
    print()

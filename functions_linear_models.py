import numpy as np
import time
import sys
#from scipy import signal, stats,special
#from scipy import interpolate as interp
#import math as math

def normalize_vec(vec):
    """ For vec of shape N x M, with N being the number of elements and M the number of coordinates"""
    vec_sum = np.sqrt(np.sum(vec**2,axis=1))
    vec_n = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        vec_n[:,i] = vec[:,i]/vec_sum

def gen_linear_mod_fit(x,y):
    #g(x) = a + bx, y = e^g(x)/(1+e^g(x))
    
    g_x = -np.log(1/y - 1)
    sx = np.std(x)
    sg = np.std(g_x)
    rgx = np.corrcoef(x,g_x)[0,1]
    
    beta = rgx*sg/sx
    alpha = np.mean(g_x)-beta*np.mean(x)
    return beta, alpha

def LinModv1 (indep_var, depen_var, w):
    ''' indep_var should be of shape n_totalxn_elem and the same for dep_var'''
    t = time.time()
    ind_var = np.zeros((indep_var.shape[0],indep_var.shape[1]+1))
    ind_var[:,0] = np.ones((indep_var.shape[0],))
    ind_var[:,1:] = indep_var
        
    dep_var = np.zeros((depen_var.shape[0],depen_var.shape[1]+1))
    dep_var[:,1:] = depen_var
    n_elem = ind_var.shape[1]
    n_total = ind_var.shape[0]
    n_bins = int(n_total/w)
    
    B_vec = np.zeros((n_bins,n_elem,n_elem))
    Corr_vec = np.zeros((n_bins,n_elem,n_elem))
    Corr_est = np.zeros((n_bins,n_elem,n_elem))
    r_vec = np.zeros((n_bins,n_elem))
    j = 0
    for i in range(n_bins):
        Cxx = np.dot(ind_var[i*w:(i+1)*w].T,ind_var[i*w:(i+1)*w])
        Cyx = np.dot(dep_var[i*w:(i+1)*w].T,ind_var[i*w:(i+1)*w])
        Corr_vec[i] = np.cov(dep_var[i*w:(i+1)*w].T)
        if np.linalg.det(Cxx) > sys.float_info.epsilon:
            iCxx = np.linalg.inv(np.nan_to_num(Cxx))
            B_vec[i] = np.dot(Cyx,iCxx)
            dep_est = np.dot(B_vec[i],ind_var[i*w:(i+1)*w].T).T
            Corr_est[i] = np.diag(np.cov(dep_est.T,dep_var[i*w:(i+1)*w].T)[:n_elem,n_elem:])
            r_vec[i,:] = np.diag(np.corrcoef(dep_est.T,dep_var[i*w:(i+1)*w].T)[:n_elem,n_elem:])
        else:
            j += 1
            continue
    print('Number of invalid windows: ', j)
    return B_vec, r_vec, Corr_vec, Corr_est

def rect_kernel(w):
    h0 = np.array([1/w for i in range(w)])
    return h0/np.sum(h0)

def gauss_kernel95(sigma):
    h = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*np.array([i/sigma for i in range(-3*sigma,3*sigma)])**2)/sigma
    return h
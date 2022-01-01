import numpy

def L1_dist(vec):
    L1_mat = np.zeros((len(vec),len(vec)))
    for i in range(len(vec)):
        for j  in range(len(vec)):
            L1_mat[i,j] = np.sum((vec[j]-vec[i])**2)**0.5
    return L1_mat

def rearrange(n, mat):
    arr_new = []
    arr_new.append(0)
    
    for i in range(0,n-3):
        if arr_new[i]+1!=n: 
                r = np.arange(0,n-1)
                     n_l = []
                for el in r:
                    if el not in arr_new:
                        n_l.append(el)
                v = np.argmin(mat[arr_new[i],n_l])
            
                arr_new.append(n_l[v])
        else:
            l = np.arange(0,n)
            n_l = []
            for el in l:
                if el not in arr_new:
                    n_l.append(el)
            v = np.argmin(mat[arr_new[i],n_l])
            
            arr_new.append(n_l[v])
      
    return arr_new

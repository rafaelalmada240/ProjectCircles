import numpy as np
import scipy.signal as sig

def length(x,y,dt):
    #Calculate the length of a parametric curve
    dx = np.diff(x)/dt
    dy = np.diff(y)/dt
    s = (dx**2+dy**2)**0.5
    return np.sum(s*dt)

def loss_function(x,y,dt):
    #Loss function for the transformation of a closed loop into a circle
    r_c = np.sqrt(x**2+y**2) #radius
    dk_tot = np.zeros(x.shape)
    L = length(x,y,dt) #Circle circumference
    dk_tot[:]=r_c[:]-L/(2*np.pi) #radius difference

    return dk_tot

def Gradient_Descent(g,dt,n_stop,prec_r,mu):
    # Gradient Descent approach using a non-stochastic approach
    f = np.array(g)
    t = np.arange(0,len(g[0]))*dt
    n_step = 0 #step counter
    peak_k = 1 #maximum radius difference
    rd_list = []
    f_list = []
    #(n_step < n_stop) and
    while (peak_k > prec_r) and (n_step < n_stop):
        
        f_0 = f #current iteration
        f_list.append(np.array(f_0))
        dk_tot = loss_function(f_0[0],f_0[1],dt)#calculate the loss function
        rd_list.append(dk_tot)
        aux = np.array([np.cos(t),np.sin(t)])#polar coordinate orientation
        f_1 = f_0 - aux*mu*dk_tot#next iteration
    
        peak_k = np.max(np.abs(dk_tot))
        for i in range(2):
            f[i] = sig.savgol_filter(f_1[i],15,3)
            
        len_r = length(f_1[0],f_1[1],dt)
        len_o = length(g[0],g[1],dt)
        f_1 = f_1*len_o/len_r
    
        n_step += 1
    print('Total number of steps: ' + str(n_step))
    print('Maximum curvature difference: '+str(peak_k))
    print('Ratio of total length: '+str(len_o/len_r))
    
    return f, f_list, rd_list
import numpy as np
import scipy.signal as sig
import functions_linear_models as flm

def length(x,y,dt):
    #Calculate the length of a parametric curve
    dx = np.diff(x)/dt
    dy = np.diff(y)/dt
    s = (dx**2+dy**2)**0.5
    return np.sum(s*dt)

def same_size_diff(f):
    diff_f = np.zeros(np.size(f))
    diff_f[:-1] = np.diff(f)
    diff_f[-1] = -(f[-2]-f[-1])
    return diff_f

def same_size_diff2(f):
    diff_f = np.zeros(np.size(f))
    diff_f[:-2] = np.diff(np.diff(f))
    diff_f[-2] = 0.5*(f[-4]+f[-2]-2*f[-3])
    diff_f[-1] = 0.5*(f[-3]+f[-1]-2*f[-2])
    return diff_f

def circular_conv(f,sigma):
    conv_f = np.zeros(f.shape)
    sigma = int(sigma) 
    conv_f[sigma*3:-3*sigma+1] = np.convolve(f[:],flm.gauss_kernel95(sigma),'valid')
    int_s = len(f)//2-(sigma*3)//2
    int_e = len(f)//2+(sigma*3)//2
    f1 = np.convolve(np.fft.fftshift(f[:]),flm.gauss_kernel95(sigma),'valid')
    f2 = np.convolve(np.fft.fftshift(f[:]),flm.gauss_kernel95(sigma),'valid')
    conv_f[:sigma*3+1] = np.fft.ifftshift(f1)[:sigma*3+1]
    conv_f[-sigma*3:] = np.fft.ifftshift(f2)[-sigma*3:]
    
    return conv_f 


def loss_function3(x,y,dt,L):
    r_c = np.sqrt(x**2+y**2) #radius
    R = L/(2*np.pi)
    #d2r = same_size_diff2(r_c)/dt**2 #diffusive flow
    #d2r = (r_c**2-R**2)/2 #potential energy driven flow
    #d2r = (r_c**2-R**2)/2 + same_size_diff2(r_c)
    d2r = (r_c-R) + same_size_diff2(r_c)
    #d2r = 2*(r_c-R)#By solving for constraint
    #d2r = (r_c/R)*(np.abs(R**2-r_c**2))**(-0.5)*same_size_diff(r_c)/dt #By solving for constraint and not replacing the derivative
    d2rc = circular_conv(d2r,np.sqrt(len(d2r)/2))

    return d2rc

def Grad_D_Dif_Eq_rad(g,dt,n_stop,prec_r,mu):
    # Gradient Descent approach using a non-stochastic approach
    f = np.array(g) #parametric curve
    t = np.arange(0,len(g[0]))*dt #time vector
    n_step = 0 #step counter
    peak_k = 1 #maximum radius difference
    r = np.sqrt(f[0]**2+f[1]**2)
    aux = np.array([np.cos(2*np.pi*t),np.sin(2*np.pi*t)]) #polar coordinate orientation
    f = np.array([r*np.cos(2*np.pi*t),r*np.sin(2*np.pi*t)])
    rd_list = []
    len_list = []
    f_list = []
    f_list.append(f)
    len_o = length(f[0],f[1],dt)
    ratio_l = 1
    len_list.append(len_o)
    
    while ((peak_k >= prec_r) and (n_step <= n_stop)) and (ratio_l <= 10):
        
        f_0 = f #current iteration
        
        
        dk_tot = loss_function3(f_0[0],f_0[1],dt,len_o)#calculate the loss function
        rd_list.append(dk_tot)
       
        
        r_0 = np.sqrt((f_0[0]**2+f_0[1]**2))
        r_1 = r_0 - mu*dk_tot*dt#next iteration
        
        f_1 = np.array([r_1*np.cos(2*np.pi*t),r_1*np.sin(2*np.pi*t)])
    
        peak_k = np.max(np.abs(dk_tot))
        #for i in range(2):
            #f[i] = circular_conv(f_1[i],np.sqrt(len(f_1[i])/2))
        #    f[i] = f_1[i]
        #    f[i] = (f[i]-np.mean(f[i]))
        
        f = f_1   
        len_r = length(f[0],f[1],dt)
        lenf1 = length(f_1[0],f_1[1],dt)
        len_list.append(len_r)
        f_list.append(np.array(f_1))
        ratio_l = len_o/len_r
        
    
        n_step += 1
    print('Total number of steps: ' + str(n_step))
    print('Maximum curvature difference: '+str(peak_k))
    print('Ratio of total length: '+str(len_o/len_r))
    
    return f, f_list, rd_list, len_list

import numpy as np
import matplotlib.pyplot as plt
import cv2 
import functions_gradient_circles as fgc

def saveframe(x_array, N_init,N_fin,f_step,dt,file_path):
    for k in range(N_init,N_fin,f_step):
        plt.figure(figsize=(6,6))
        plt.xlabel('x')
        plt.ylabel('y')
    
        plt.plot(x_array[0,0],x_array[0,1],'k--',alpha=0.4)
        Len = fgc.length(x_array[k,0],x_array[k,1],dt)
        Len_0 =fgc.length(x_array[0,0],x_array[0,1],dt) 
        plt.plot(Len_0/Len*x_array[k,0],Len_0/Len*x_array[k,1],'b.',alpha=0.4)
        Mx = np.max(np.abs(Len_0/Len*x_array[k,0]))
        My = np.max(np.abs(Len_0/Len*x_array[k,1]))
        plt.xlim(-1.01*Mx,1.01*Mx)
        plt.ylim(-1.01*My,1.01*My)
        plt.grid
    

        #plt.legend(['m1','m2','M'])
        plt.savefig(file_path+str(k)+'.jpg')
        plt.close()
    return

def loadframe(N_init,N_fin,f_step,file_path):
    img_array = []
    #file_path = g
    for k in range(N_init,N_fin,f_step):
        img = cv2.imread(file_path + str(k)+'.jpg')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    return img_array,size
 
def savevideo(img_array,filename,size):
    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return

def makemovie(x_array,dt):
    N_initial = int(input('Starting frame: '))
    N_final = int(input('Stopping frame: '))
    fr_step = int(input('Frame step: '))
    file_path = input('Image File Path: ')
    filename = input('Video file name (with .mp4 included): ')
    
    saveframe(x_array, N_initial,N_final,fr_step,dt,file_path)
    img_array,size = loadframe(N_initial,N_final,fr_step,file_path)
    savevideo(img_array,filename,size)
    return
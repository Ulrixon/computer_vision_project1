#%% import pakages
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/moon.png",0)
plt.imshow(img, cmap="gray")
# %% low_pass_filter

def low_pass_filter(img):
    lap=np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])

    lapimage= np.zeros(img.shape)
    im = np.pad(img.copy(), ((lap.shape[0]//2, lap.shape[0]//2), (lap.shape[0]//2, lap.shape[0]//2)), mode = "edge")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            acc=0
            for ki in range(lap.shape[0]):
                for kj in range(lap.shape[1]):
                    acc+=im[range(i,i+lap.shape[0]),:][:,range(j,j+lap.shape[1])][ki,kj]*lap[ki,kj]
            lapimage[i,j]=acc        
        
    plt.imshow(lapimage, cmap="gray")                
    
    return lapimage

#lrlapimgaes=(lapimage-np.min(lapimage))/(np.max(lapimage.max()) - np.min(lapimage.min))
newimg=low_pass_filter(img)
#plt.imshow(lapimage, cmap="gray") 
plt.imshow(newimg, cmap="gray")

#%%guassian_smoothing
def guassian_smoothing(img):
    lap=np.array([[1, 2 ,1],[ 2, 4 ,2] ,[1 ,2 ,1]])/16
    lapimage= np.zeros(img.shape)
    im = np.pad(img.copy(), ((lap.shape[0]//2, lap.shape[0]//2), (lap.shape[0]//2, lap.shape[0]//2)), mode = "edge")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            acc=0
            for ki in range(lap.shape[0]):
                for kj in range(lap.shape[1]):
                    acc+=im[range(i,i+lap.shape[0]),:][:,range(j,j+lap.shape[1])][ki,kj]*lap[ki,kj]
            lapimage[i,j]=acc

    plt.imshow(lapimage, cmap="gray") 
    return lapimage

newimg=guassian_smoothing(img)
#plt.imshow(lapimage, cmap="gray") 
plt.imshow(newimg, cmap="gray")
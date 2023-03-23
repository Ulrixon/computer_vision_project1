#%% import pakages
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%%
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/moon.png",0)
plt.imshow(img, cmap="gray")
# %% laplacian sharpening gray picture

def laplacian_sharpening_gray(img):
    #lap=np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    lap=np.array([[-1, -1 ,-1],[ -1, 8 ,-1] ,[-1 ,-1 ,-1]])
    lapimage= np.zeros(img.shape)
    im = np.pad(img.copy(), ((lap.shape[0]//2, lap.shape[0]//2), (lap.shape[0]//2, lap.shape[0]//2)), mode = "edge")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            acc=0
            for ki in range(lap.shape[0]):
                for kj in range(lap.shape[1]):
                    acc+=im[range(i,i+lap.shape[0]),:][:,range(j,j+lap.shape[1])][ki,kj]*lap[ki,kj]
            lapimage[i,j]=acc        
        
    #plt.imshow(lapimage, cmap="gray")                
    newimg=np.clip(img+lapimage,0,255).astype(np.uint8)
    return newimg

#lrlapimgaes=(lapimage-np.min(lapimage))/(np.max(lapimage.max()) - np.min(lapimage.min))
newimg=laplacian_sharpening_gray(img)
#plt.imshow(lapimage, cmap="gray") 
plt.imshow(newimg, cmap="gray")   

# %% laplacian sharpening color picture
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/lenna.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

def laplacian_sharpening_color(img):

    return [laplacian_sharpening_gray(img[:,:,0]),laplacian_sharpening_gray(img[:,:,1])
        ,laplacian_sharpening_gray(img[:,:,2])]
newimg=laplacian_sharpening_color(img)
newimg= np.array(newimg)
newimg=np.transpose(newimg, (1, 2, 0))
plt.imshow(newimg)
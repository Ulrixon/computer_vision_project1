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
cv2.imwrite("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/average_moon.png",newimg)
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
cv2.imwrite("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/gaussian_moon.png",newimg)
plt.imshow(newimg, cmap="gray")
# %% color average filter


img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/lenna.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
def average_smoothing_color(img):
    return np.transpose(np.array([low_pass_filter(img[:,:,0]),low_pass_filter(img[:,:,1]),low_pass_filter(img[:,:,2])]), (1, 2, 0)).astype('uint8')
#newimg=[HE(img[:,:,0]),img[:,:,1],img[:,:,2]]
newimg=average_smoothing_color(img)
plt.imshow(newimg)
rgbimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
cv2.imwrite("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/average_lenna.png",rgbimg)

# %%


img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/lenna.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
def guassian_smoothing_color(img):
    return np.transpose(np.array([guassian_smoothing(img[:,:,0]),guassian_smoothing(img[:,:,1]),guassian_smoothing(img[:,:,2])]), (1, 2, 0)).astype('uint8')
#newimg=[HE(img[:,:,0]),img[:,:,1],img[:,:,2]]
newimg=guassian_smoothing_color(img)
plt.imshow(newimg)
rgbimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
cv2.imwrite("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/gaussian_lenna.png",rgbimg)
# %%

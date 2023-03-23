#%% import pakages
import cv2
import matplotlib.pyplot as plt
import numpy as np
# %% read img
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/KneYW.jpg",0)
plt.imshow(img, cmap="gray")
# %%
plt.hist(img.flatten(),bins=256)
plt.show()
# %% histogram eq for gray picture
def HE(img,hist_plot_show=False):
    L=256
    counting= np.zeros(L)

    for i in img: # count gray level
        for j in i:
            counting[j]+=1

    
    pdf=counting/sum(counting)

    cdf=np.zeros(L)
    a=0
    for i in range(L): # generate cdf
        a+=pdf[i]
        cdf[i]=a
    newCounting=np.round(cdf*(L-1))

    newimg= np.zeros(img.shape)
    for i in range(L):
        newimg[img==i]=newCounting[i]
    if hist_plot_show==True:
        plt.hist(newimg.flatten(),bins=256)
        plt.show()
    return newimg

#%%
newimg= HE(img)
plt.imshow(newimg, cmap="gray")
#%% HE_color
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/LLFlow-main/images/test_images/sky_and_mount.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
plt.imshow(img)
def HE_color(img):
    return cv2.cvtColor(np.transpose(np.array([HE(img[:,:,0]),img[:,:,1],img[:,:,2]]), (1, 2, 0)).astype('uint8'),cv2.COLOR_YCrCb2BGR)
newimg=[HE(img[:,:,0]),img[:,:,1],img[:,:,2]]
newimg=HE_color(img)
#newimg= np.array(newimg)
#newimg=np.transpose(newimg, (1, 2, 0))
newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
plt.imshow(newimg)
#%%
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/LLFlow-main/images/sky_and_mount.jpg",cv2.IMREAD_COLOR)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# %% adaptive HE
wsz=51
im = np.pad(img.copy(), ((wsz//2, wsz//2), (wsz//2, wsz//2)), mode = 'reflect')
plt.imshow(im, cmap="gray")
#%% time_consuming AHE
newimg= np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        newimg[i,j]=HE(im[range(wsz+i)][:,range(wsz+j)])[wsz//2+i,wsz//2+j]
plt.imshow(newimg, cmap="gray")


# %%


def center_pixel_he(img,wsz,x,y):
    L=256
    counting= np.zeros(L)

    for i in img: # count gray level
        for j in i:
            counting[j]+=1

    
    pdf=counting/sum(counting)

    cdf=np.zeros(L)
    a=0
    for i in range(L): # generate cdf
        a+=pdf[i]
        cdf[i]=a
    newCounting=np.round(cdf*(L-1))

    
    new_pixel_value=newCounting[img[wsz//2+x,wsz//2+y]]

    return new_pixel_value

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        newimg[i,j]=center_pixel_he(im[range(wsz+i)][:,range(wsz+j)],wsz,i,j)

from multiprocessing import Pool

    



with Pool(4) as pool:
    newimg=pool.map(center_pixel_he, [wsz, range(img.shape[0]), range(img.shape[1])])
plt.imshow(newimg, cmap="gray")
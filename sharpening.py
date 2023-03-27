#%% import pakages
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%%
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/moon.png",0)
plt.imshow(img, cmap="gray")
# %% laplacian sharpening gray picture

def laplacian_sharpening_gray(img,ShF=100): #shf is sharpening factor
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
    #lapimage=((lapimage-np.min(lapimage))*(1/(np.max(lapimage)-np.min(lapimage)))*255).astype('uint8')
    
    lapimage= lapimage*ShF/np.amax(lapimage)
    plt.imshow(lapimage, cmap="gray")
    newimg=img+lapimage                
    newimg=np.clip(newimg,0,255).astype(np.uint8)
    #((newimg-np.min(newimg))*(1/(np.max(newimg)-np.min(newimg)))*255).astype('uint8')
    #
    
    return newimg

#lrlapimgaes=(lapimage-np.min(lapimage))/(np.max(lapimage.max()) - np.min(lapimage.min))
newimg=laplacian_sharpening_gray(img,255)
#plt.imshow(lapimage, cmap="gray") 
plt.imshow(newimg, cmap="gray")   

# %% laplacian sharpening color picture
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/lenna.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
#%%
def laplacian_sharpening_color(img):

    return [laplacian_sharpening_gray(img[:,:,0]),laplacian_sharpening_gray(img[:,:,1])
        ,laplacian_sharpening_gray(img[:,:,2])]
newimg=laplacian_sharpening_color(img)
newimg= np.array(newimg)
newimg=np.transpose(newimg, (1, 2, 0))
plt.imshow(newimg)
# %% sobel with guassian snoothing sharpening


#ShF=255

def sobel_edge_gray(img,vertical=False):
    #lap=np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])\
    if( vertical==True):
        lap=np.array([[1, 2 ,1],[ 0, 0 ,0] ,[-1 ,-2 ,-1]])
    else:

        lap=np.array([[-1, 0 ,1],[ -2, 0 ,2] ,[-1 ,0 ,1]])
    lapimage= np.zeros(img.shape)
    im = np.pad(img.copy(), ((lap.shape[0]//2, lap.shape[0]//2), (lap.shape[0]//2, lap.shape[0]//2)), mode = "edge")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            acc=0
            for ki in range(lap.shape[0]):
                for kj in range(lap.shape[1]):
                    acc+=im[range(i,i+lap.shape[0]),:][:,range(j,j+lap.shape[1])][ki,kj]*lap[ki,kj]
            lapimage[i,j]=acc        
        
    #lapimage=((lapimage-np.min(lapimage))*(1/(np.max(lapimage)-np.min(lapimage)))*255).astype('uint8') # scale to 0-255
    #lapimage=np.clip(lapimage,0,255).astype(np.uint8)
    #plt.imshow(lapimage, cmap="gray") 
    return lapimage



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

    #plt.imshow(lapimage, cmap="gray") 
    return lapimage


def sobel_gaussian_sharpening(img,ShF=255):
    filterx=sobel_edge_gray(img)

    filtery=sobel_edge_gray(img,True)
    filterx=guassian_smoothing(filterx)
    filtery=guassian_smoothing(filtery)

    lapimage=(filterx+filtery)*ShF/np.amax(filterx+filtery)
    newimg=img+lapimage                
    newimg=np.clip(newimg,0,255).astype(np.uint8)
    return newimg

newimg=sobel_gaussian_sharpening(img,255)
plt.imshow(newimg, cmap="gray")

#%% color sobel_gaussian_sharpening
def sobel_gaussian_sharpening_color(img):

    return [sobel_gaussian_sharpening(img[:,:,0]),sobel_gaussian_sharpening(img[:,:,1])
        ,sobel_gaussian_sharpening(img[:,:,2])]
newimg=sobel_gaussian_sharpening_color(img)
newimg= np.array(newimg)
newimg=np.transpose(newimg, (1, 2, 0))
plt.imshow(newimg)


# %%

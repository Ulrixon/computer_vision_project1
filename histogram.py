#%% import pakages
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from random import randrange\
import itertools as it
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
#newimg=[HE(img[:,:,0]),img[:,:,1],img[:,:,2]]
newimg=HE_color(img)
#newimg= np.array(newimg)
#newimg=np.transpose(newimg, (1, 2, 0))
newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
plt.imshow(newimg)
#%%
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/LLFlow-main/images/sky_and_mount.jpg",cv2.IMREAD_COLOR)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# %% adaptive HE not recommend using this, u can use ahe by clahe chuck set enableclip=False
wsz=51
im = np.pad(img.copy(), ((wsz//2, wsz//2), (wsz//2, wsz//2)), mode = 'reflect')
plt.imshow(im, cmap="gray")
#%% time_consuming AHE
newimg= np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        newimg[i,j]=HE(im[range(wsz+i)][:,range(wsz+j)])[wsz//2+i,wsz//2+j]
plt.imshow(newimg, cmap="gray")


# %% CLAHE grayscale

 #8x8

def clahe_grayscale(img,ClipLimits=8,tile=20,enableclip=True): # tile should be even number

    im = np.pad(img.copy(), ((0, tile-img.shape[0]%tile), (0, tile-img.shape[1]%tile)), mode = 'reflect')
    L=256
    newim=np.zeros(im.shape)
    hist=np.zeros((im.shape[0]//tile,im.shape[1]//tile,L),)
    for i in range(im.shape[0]//tile): # for each tile do HE and record transform fun. 
        for j in range(im.shape[1]//tile):
            subimg= im[i*tile:i*tile+tile,j*tile:j*tile+tile]
        
            counting= np.zeros(L)

            for k in subimg: # count gray level
                for l in k:
                    counting[l]+=1

            if(enableclip==True): #enable clip or not ,if not then it is ahe

                clip=sum(counting[counting>ClipLimits]-ClipLimits)
                counting[counting>ClipLimits]=ClipLimits
        
                pdf=(counting+clip/256)/sum(counting+clip/256)
            else:
                pdf=counting/sum(counting)
            cdf=np.zeros(L)
            a=0
            for n in range(L): # generate cdf
                a+=pdf[n]
                cdf[n]=a
            newCounting=np.round(cdf*(L-1))
            hist[i,j,:]=newCounting
            newimg= np.zeros(subimg.shape)
            for o in range(L):
                newimg[subimg==o]=newCounting[o]
            newim[i*tile:i*tile+tile,j*tile:j*tile+tile]=newimg



#plt.imshow(newim[0:img.shape[0],0:img.shape[1]], cmap="gray")

# interpolation part
    def interpolation(hist,i,j,subimg,tileshapex,tileshapey):
        newsubimg=np.zeros(subimg.shape)

        for k in range(subimg.shape[0]): # x-axis top
            inversek= subimg.shape[0]-k-0.5
            for l in range(subimg.shape[1]): #y-axis left
                inversel= subimg.shape[1]-l-0.5
                newsubimg[k,l]=((hist[0,int(subimg[k,l])]*(inversek)+ hist[2,int(subimg[k,l])]*(k+0.5))*(inversel)/subimg.shape[0]
                +(hist[1,int(subimg[k,l])]*(inversek )+hist[3,int(subimg[k,l])]*(k+0.5))/subimg.shape[0]*(l+0.5))/subimg.shape[1]
                #print(newsubimg[k,l])

        return newsubimg
    afterinterim=np.zeros(im.shape)
    for i in range(0,im.shape[0]//tile-1): # for each purple square tile do bilinear interpolation 
        for j in range(0,im.shape[1]//tile-1):
        
            subhist=np.array([hist[i,j,:],hist[i,j+1,:],hist[i+1,j,:],hist[i+1,j+1,:]])
        
            afterinterim[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)]=interpolation(subhist,
                i,j,im[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)],tile,tile)

    for i in it.chain(range(0,round(tile/2)),range(im.shape[0]-round(tile/2),im.shape[0])): # for each cornor half tile do he 
        for j in it.chain(range(0,round(tile/2)),range(im.shape[1]-round(tile/2),im.shape[1])):
            afterinterim[i,j]=newim[i,j]

    for i in range(0,im.shape[0]//tile-1):
        for j in [0,im.shape[1]//tile-1]:  # for left right edge half tile do linear interpolate
            subhist=np.array([hist[i,j,:],hist[i,j,:],hist[i+1,j,:],hist[i+1,j,:]])
            if(j==0):
            

                afterinterim[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile):int(j*tile+1/2*tile)]=interpolation(subhist,
                i,j,im[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile):int(j*tile+1/2*tile)],tile,tile)
            else:
                afterinterim[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile)]=interpolation(subhist,
                i,j,im[int(i*tile+1/2*tile):int(i*tile+tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile)],tile,tile)

    
    for i in [0,im.shape[0]//tile-1]:
        for j in range(0,im.shape[1]//tile-1):  # for top bottom edge half tile do linear interpolate
            subhist=np.array([hist[i,j,:],hist[i,j+1,:],hist[i,j,:],hist[i,j+1,:]])
            if(i==0):
            

                afterinterim[int(i*tile):int(i*tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)]=interpolation(subhist,
                i,j,im[int(i*tile):int(i*tile+1/2*tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)],tile,tile)
            else:
                afterinterim[int(i*tile+1/2*tile):int(i*tile+tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)]=interpolation(subhist,
                i,j,im[int(i*tile+1/2*tile):int(i*tile+tile),
                int(j*tile+1/2*tile):int(j*tile+tile+1/2*tile)],tile,tile)


    #plt.imshow(afterinterim[0:img.shape[0],0:img.shape[1]], cmap="gray")
    return afterinterim[0:img.shape[0],0:img.shape[1]]

afterinterim= clahe_grayscale(img)
plt.imshow(afterinterim, cmap="gray")

#%% clahe_color
img = cv2.imread("/mnt/c/Users/ryan7/Documents/GitHub/computer_vision_project1/LLFlow-main/images/test_images/sky_and_mount.jpg",cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
plt.imshow(img)
def CLAHE_color(img):
    return cv2.cvtColor(np.transpose(np.array([clahe_grayscale(img[:,:,0]),img[:,:,1],img[:,:,2]]), (1, 2, 0)).astype('uint8'),cv2.COLOR_YCrCb2BGR)
#newimg=[HE(img[:,:,0]),img[:,:,1],img[:,:,2]]
newimg=CLAHE_color(img)
#newimg= np.array(newimg)
#newimg=np.transpose(newimg, (1, 2, 0))
newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
plt.imshow(newimg)



# %%
clahe = cv2.createCLAHE(clipLimit = 8)
final_img = clahe.apply(img)
plt.imshow(final_img, cmap="gray")
# %%

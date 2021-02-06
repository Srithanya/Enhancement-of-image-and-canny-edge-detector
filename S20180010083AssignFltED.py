import cv2
import os
import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#-------------------------------------------- Question a ---------------------------------------------------------#

kernel_size = 3
scale = 1
delta = 0
ddepth = cv2.CV_16S
img = cv2.imread('K Sai Sri Thanya 083 - Sample_flt.tif')

#Laplacian image of a
lap_image= cv2.Laplacian(img, cv2.CV_64F, ksize=3)
cv2.imshow('laplacian',lap_image)
cv2.waitKey(0)


#Sharpened image by adding a and b
img1=np.uint8(np.absolute(img))
img2 = np.uint8(np.absolute(lap_image))
sharpened_image1 = cv2.add(img1,img2)
cv2.imshow('sharpened image',sharpened_image1)
cv2.waitKey(0)

#Sobel gradient of a
sobelx_grad = cv2.Sobel(img,cv2.CV_64F,1,0)  
sobely_grad= cv2.Sobel(img,cv2.CV_64F,0,1) 
absx= np.uint8(np.absolute(sobelx_grad))
absy =np.uint8(np.absolute(sobely_grad))
sobel = cv2.bitwise_or(absx,absy)
cv2.imshow('sobel gradient',sobel)
cv2.waitKey(0)

#Sobel image smoothd by 5x5 filter
kernel = np.ones((5,5),np.float32)/25
smoothed_image = cv2.filter2D(sobel,-1,kernel)
cv2.imshow('smoothed sobel',smoothed_image)
cv2.waitKey(0)

#Mask image by product of c and e
sharp = np.uint8(np.absolute(sharpened_image1))
smooth = np.uint8(np.absolute(smoothed_image))
mask_image=cv2.bitwise_and(sharp,smooth)
cv2.imshow('mask',mask_image)
cv2.waitKey(0)


#Sharpened image by sum of a and f
sharpened_image2=cv2.add(img,mask_image)
titles = ['Image', 'Sharpened image']
images = [img,sharpened_image2]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
cv2.waitKey(0)


#Final result by applying power law tarnsformation
gamma=0.5
final_image = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
titles = ['Image', 'Final image']
images = [img,final_image]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
cv2.waitKey(0)


    
#--------------------------------------------------Question b----------------------------------------------#


def Sobel(img):
    c=np.zeros(img.shape)
    c1=np.zeros(img.shape)
    x_grad=np.zeros(img.shape)
    y_grad=np.zeros(img.shape)
    size=img.shape
    k_x=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    k_y=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            x_grad[i,j]=np.sum(np.multiply(img[i-1:i+2,j-1:j+2],k_x))
            y_grad[i,j]=np.sum(np.multiply(img[i-1:i+2,j-1:j+2],k_y))
    
    c=np.sqrt(np.square(x_grad)+np.square(y_grad))
    c=np.multiply(c,255.0/c.max())

    angles_array=np.rad2deg(np.arctan2(y_grad,x_grad))
    angles_array[angles_array<0]+=180
    c=c.astype('uint8')
    return c,angles_array


def non_maximum_supp(img,angles_array):
    size=img.shape
    s=np.zeros(size)
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            if (0<=angles_array[i,j]<22.5) or (157.5<=angles_array[i,j]<=180):
                val=max(img[i,j-1],img[i,j+1])
            elif (22.5<=angles_array[i,j]<67.5):
                val=max(img[i-1,j-1],img[i+1,j+1])
            elif (67.5<=angles_array[i,j]<112.5):
                val=max(img[i-1,j],img[i+1,j])
            else:
                val=max(img[i+1,j-1],img[i-1,j+1])
            
            if img[i,j]>=val:
                s[i,j]=img[i,j]
    s=np.multiply(s,255.0/s.max())
    return s

    
def double_thresholding_hysteresis(img,l,h):
    size=img.shape
    res=np.zeros(size)
    w1,w2=np.where((img>l)&(img<=h))
    s1,s2=np.where(img>=h)
    weak=50
    strong=255
    res[s1,s2]=strong
    res[w1,w2]=weak
    x_arr=np.array((-1,-1,0,1,1,1,0,-1))
    y_arr=np.array((0,1,1,1,0,-1,-1,-1))
    size=img.shape
    while len(s1):
        a=s1[0]
        b=s2[0]
        s1=np.delete(s1,0)
        s2=np.delete(s2,0)
        for i in range(len(x_arr)):
            x1=a+x_arr[i]
            y1=b+y_arr[i]
            if((x1>=0& x1<size[0] & y1>=0 & y1<size[1])):
                if (res[x1,y1]==weak):
                    res[x1,y1]=strong
                    np.append(s1,x1)
                    np.append(s2,y1)
    res[res!=strong]=0
    return res


def Canny_detection(image,l,h):
    img=cv2.GaussianBlur(image,(3,3),0)
    image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image,angles=Sobel(image)
    image=non_maximum_supp(image,angles)
    grad=np.copy(image)
    image=double_thresholding_hysteresis(image,l,h)
    return image,grad

if __name__ == "__main__":
    image=mpimg.imread('K Sai Sri Thanya 083 - lena.gif',cv2.IMREAD_COLOR)
    image,grad=Canny_detection(image, 0, 50)
    plt.imshow(image,cmap='gray')
    plt.show()
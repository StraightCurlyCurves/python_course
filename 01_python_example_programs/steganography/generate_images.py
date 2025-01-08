import numpy as np
import cv2 as cv

# generate an image with a black background
img = np.zeros((200,300,3), dtype=np.uint16)

# copy the original black image 
img_clear = img.copy()
img_clear[:,:,:] = 127*255

img_clear_noise = img.copy()
# noise = np.random.randint(0, 255**2, img_clear_noise.shape[:2])
noise = np.random.randint(0, 255**2, img_clear_noise.shape)
img_clear_noise[:,:,:] = noise
# img_clear_noise[:,:,0] = noise
# img_clear_noise[:,:,1] = noise
# img_clear_noise[:,:,2] = noise


img_secret = img.copy()
img_secret[:,:,2] = 255*255
img_secret[40:160, 130:170, :] = 255*255
img_secret[80:120, 90:210, :] = 255*255

cv.imwrite("img_clear.png", img_clear)
cv.imwrite("img_clear_noise.png", img_clear_noise)
cv.imwrite("img_secret.png", img_secret)
# cv.imwrite("img_secret_2.png", img_secret2)
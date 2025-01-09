import numpy as np
import cv2 as cv

# generate an image with a black background
img = np.zeros((200,300,3), dtype=np.uint8)

# create the visible image that will be used to hide the secret image
img_public = img.copy()
img_public[:,:,:] = 127 # gray 16-bit image

# create a visible image with noise
img_public_noise = np.random.randint(0, 255, img.shape)

# create the secret image that will be hidden in the visible image
img_private = img.copy()
img_private[:,:,2] = 255
img_private[40:160, 130:170, :] = 255
img_private[80:120, 90:210, :] = 255

# save the images
cv.imwrite("img_public.png", img_public)
cv.imwrite("img_public_noise.png", img_public_noise)
cv.imwrite("img_private.png", img_private)
import numpy as np
import cv2 as cv

def encrypt(img_clear: np.ndarray, img_secret: np.ndarray):
    assert img_clear.shape == img_secret.shape
    assert img_clear.dtype == np.uint8
    assert img_secret.dtype == np.uint8

    img_encrypted = img_secret.astype(np.uint16) | img_clear.astype(np.uint16) << 8
    return img_encrypted

def decrypt(img: np.ndarray):
    assert img.dtype == np.uint16

    img_clear = (img >> 8).astype(np.uint8)
    img_secret = img.astype(np.uint8)
    return img_secret, img_clear

def load_8bit_image(filename):
    return cv.imread(filename)

def load_16bit_img(filename):
    return cv.imread(filename, -1)

def save_image(filename, img):
    cv.imwrite(filename, img)

if __name__ == '__main__':
    img_clear = load_8bit_image('img_clear.png')
    img_clear_noise = load_8bit_image('img_clear_noise.png')
    img_secret = load_8bit_image('img_secret.png')

    img_encrypted = encrypt(img_clear, img_secret)
    img_encrypted_with_noise = encrypt(img_clear_noise, img_secret)

    save_image('img_encrypted.png', img_encrypted)
    save_image('img_encrypted_with_noise.png', img_encrypted_with_noise)

    img_encrypted = load_16bit_img('img_encrypted.png')
    img_encrypted_with_noise = load_16bit_img('img_encrypted_with_noise.png')

    img_decrypted, _ = decrypt(img_encrypted)
    img_decrypted_with_noise, _ = decrypt(img_encrypted_with_noise)

    save_image('img_decrypted.png', img_decrypted)
    save_image('img_decrypted_with_noise.png', img_decrypted_with_noise)


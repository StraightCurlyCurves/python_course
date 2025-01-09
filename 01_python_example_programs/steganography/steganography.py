import numpy as np
import cv2 as cv

def encrypt(img_public: np.ndarray, img_private: np.ndarray) -> np.ndarray:
    assert img_public.shape == img_private.shape
    assert img_public.dtype == np.uint8
    assert img_private.dtype == np.uint8

    img_encrypted = img_private.astype(np.uint16) | img_public.astype(np.uint16) << 8
    return img_encrypted

def decrypt(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert img.dtype == np.uint16

    img_public = (img >> 8).astype(np.uint8)
    img_private = img.astype(np.uint8)
    return img_private, img_public

def load_image(filename: str) -> np.ndarray:
    return cv.imread(filename, cv.IMREAD_UNCHANGED)

def save_image(filename: str, img) -> None:
    cv.imwrite(filename, img)

if __name__ == '__main__':
    img_public = load_image('img_public.png')
    img_public_noise = load_image('img_public_noise.png')
    img_private = load_image('img_private.png')

    img_encrypted = encrypt(img_public, img_private)
    img_encrypted_with_noise = encrypt(img_public_noise, img_private)

    save_image('img_encrypted.png', img_encrypted)
    save_image('img_encrypted_with_noise.png', img_encrypted_with_noise)

    img_encrypted = load_image('img_encrypted.png')
    img_encrypted_with_noise = load_image('img_encrypted_with_noise.png')

    img_decrypted, _ = decrypt(img_encrypted)
    img_decrypted_with_noise, _ = decrypt(img_encrypted_with_noise)

    save_image('img_decrypted.png', img_decrypted)
    save_image('img_decrypted_with_noise.png', img_decrypted_with_noise)

import numpy as np
import cv2 as cv

WINDOW_NAME = 'Mandelbrot Set'
HEIGHT, WIDTH = 800, 800

def iterations_in_mandelbrot_set(c, z, max_iter):
    for i in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return i
    return max_iter

def mandelbrot_set(width, height, x_min=None, x_max=None, y_min=None, y_max=None):
    mandelbrot = np.zeros((height, width), dtype=np.uint8)
    z = 0
    max_iter = 50

    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max, y_min, y_max = -2.0, 2.0, height / width * -2.0, height / width * 2.0

    for x in range(width):
        for y in range(height):
            c = complex(x / width * (x_max - x_min) + x_min,
                         y / height * (y_max - y_min) + y_min)
            iterations = iterations_in_mandelbrot_set(c, z, max_iter)
            if True:
                brightness = 0 if iterations == max_iter else int(min(iterations, max_iter) / max_iter * 255)
            else:
                brightness = int(min(iterations, max_iter) / max_iter * 255)
            mandelbrot[y, x] = brightness
        if x % 50 == 0:
            cv.imshow(WINDOW_NAME, mandelbrot)
            cv.waitKey(1)
    return mandelbrot

if __name__ == '__main__':
    window = cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    img = mandelbrot_set(HEIGHT, WIDTH)
    cv.imshow(WINDOW_NAME, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
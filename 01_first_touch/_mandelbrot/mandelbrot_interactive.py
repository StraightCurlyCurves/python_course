import threading
from time import perf_counter
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2 as cv

class Mandelbrot:

    def __init__(self, height: int = 480, width: int = 640, max_iter: int = 50):
        self.invert_in_set_color = True

        self.__height = height
        self.__width = width
        self._init_coordinates = -2.0, 2.0, height / width * -2.0, height / width * 2.0
        self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
        self._max_iter = max_iter
        self._z = 0
        self._columns_per_update = 50

        self._img = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)

        self._window_name = 'Mandelbrot Set'
        self._window = cv.namedWindow(self._window_name)

        cv.setMouseCallback(self._window_name, self._mouse_callback)

        self._drawing_rectangle_event = threading.Event()
        self._calculate_mandelbrot_set_abort_flag = threading.Event()
        self._is_calculating_event = threading.Event()
        self._rect_init_x, self._rect_init_y = 0, 0
        self._rect_dx, self._rect_dy = 0, 0

        self._middle_mousebutton_states = ['change_max_iter', 'change_resolution_width', 'change_resolution_height']
        self.middle_mousebutton_state = 0

    def run(self):
        self._calculate_mandelbrot_set()
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                Path('images').mkdir(parents=True, exist_ok=True)
                date_and_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'images/mandelbrot_{date_and_time}.png'
                cv.imwrite(filename, self._img)
                print(f'Image saved: {filename}')
            cv.imshow(self._window_name, self._img)
        cv.destroyAllWindows()

    @staticmethod
    def iterations_in_mandelbrot_set(c, z, max_iter):
        """
        Calculates the number of iterations required for a complex number `c` to escape the Mandelbrot set.

        Parameters:
        - c (complex): The complex number to be tested.
        - z (complex): The initial value of `z` in the iteration.
        - max_iter (int): The maximum number of iterations to perform.

        Returns:
        - int: The number of iterations required for `c` to escape the Mandelbrot set, or `max_iter` if it doesn't escape.

        """
        for i in range(max_iter):
            z = z**2 + c
            if abs(z) > 2:
                return i
        return max_iter

    def _calculate_mandelbrot_set(self, identification=0, verbose=False):
        self._is_calculating_event.set()
        self._calculate_mandelbrot_set_abort_flag.clear()
        self._img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv.imshow(self._window_name, self._img)
        cv.waitKey(1)
        global_t_0 = perf_counter()
        t_0 = perf_counter()        
        for x in range(self.__width):
            for y in range(self.__height):
                if self._calculate_mandelbrot_set_abort_flag.is_set():
                    self._is_calculating_event.clear()
                    return
                c = complex(x / self.__width * (self._re_max - self._re_min) + self._re_min,
                            y / self.__height * (self._im_max - self._im_min) + self._im_min)
                iterations = self.iterations_in_mandelbrot_set(c, self._z, self._max_iter)
                if self.invert_in_set_color:
                    brightness = 0 if iterations == self._max_iter else int(iterations / self._max_iter * 255)
                else:
                    brightness = int(iterations / self._max_iter * 255)
                self._img[y, x, :] = [brightness, brightness, brightness]
            if perf_counter() - t_0 > 0.1:
                t_0 = perf_counter()
                if verbose:
                    print(f'{identification} img address: {id(self._img)}')
                # print progress bar
                print(f'\r{" " * 50}\r{(x+1)/self.__width*100:.2f}%', end='')
                cv.imshow(self._window_name, self._img)
                cv.waitKey(1)                
        cv.imshow(self._window_name, self._img)
        print(f'\r{" " * 50}\r100% ({perf_counter() - global_t_0:.2f}s)')
        self._is_calculating_event.clear()
        return

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self._calculate_mandelbrot_set_abort_flag.set()
            self._drawing_rectangle_event.set()
            self._rect_init_x, self._rect_init_y = x, y
            self._rect_dx, self._rect_dy = 0, 0

        elif event == cv.EVENT_MOUSEMOVE:
            if self._drawing_rectangle_event.is_set():
                self._draw_rectangle(x, y)            

        elif event == cv.EVENT_RBUTTONUP:
            if self._drawing_rectangle_event.is_set():
                self._drawing_rectangle_event.clear()
                cv.imshow(self._window_name, self._img)
            else:
                self._calculate_mandelbrot_set_abort_flag.set() #TODO: problem: will not abort since initial function call gets paused
                self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
                self._calculate_mandelbrot_set(identification='RESET', verbose=False)
                cv.imshow(self._window_name, self._img)

        elif event == cv.EVENT_LBUTTONUP:
            if self._drawing_rectangle_event.is_set():
                self._drawing_rectangle_event.clear()
                if self._rect_dy != 0 and self._rect_dx != 0:
                    self._update_coordinates_from_rectangle()
                self._calculate_mandelbrot_set(identification='RECTANGLE', verbose=False)
                cv.imshow(self._window_name, self._img)

        elif event == cv.EVENT_MOUSEWHEEL:
            if self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_max_iter':
                if flags > 0:
                    self._max_iter += 10
                else:
                    self._max_iter -= 10
                print(f'max_iter: {self._max_iter}')

            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_resolution_height':
                if flags > 0:
                    self.height += 100 # This will call the height setter and adjust coordinates accordingly
                else:
                    self.height -= 100 # This will call the height setter and adjust coordinates accordingly
                print(f'height: {self.height}')
            
            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_resolution_width':
                if flags > 0:
                    self.width += 100 # This will call the width setter and adjust coordinates accordingly
                else:
                    self.width -= 100 # This will call the width setter and adjust coordinates accordingly
                print(f'width: {self.width}')

        elif event == cv.EVENT_MBUTTONUP:
            self.middle_mousebutton_state += 1
            self.middle_mousebutton_state %= len(self._middle_mousebutton_states)
            print(f'middle_mousebutton_state: {self._middle_mousebutton_states[self.middle_mousebutton_state]}') 
        
    def _draw_rectangle(self, x, y):
        img_copy = self._img.copy()
        self._rect_dx = x - self._rect_init_x
        self._rect_dy = y - self._rect_init_y
        if self._rect_dx == 0 or self._rect_dy == 0:
            return
        # calculate width and height of rectangle according to image ratio
        ratio_image = abs(self.__width / self.__height)
        ratio_draw = abs(self._rect_dx / self._rect_dy)
        if ratio_draw < ratio_image:
            self._rect_dx = int(abs(self._rect_dy) * ratio_image) * np.sign(self._rect_dx)
        else:
            self._rect_dy = int(abs(self._rect_dx) / ratio_image) * np.sign(self._rect_dy)
        cv.rectangle(img_copy, (self._rect_init_x, self._rect_init_y), (self._rect_init_x + self._rect_dx, self._rect_init_y + self._rect_dy), (0, 0, 255), 1)
        cv.imshow(self._window_name, img_copy)

    def _update_coordinates_from_rectangle(self):
        old_re_min, old_re_max, old_im_min, old_im_max = self._re_min, self._re_max, self._im_min, self._im_max
        self._re_p1 = self._rect_init_x / self.__width * (old_re_max - old_re_min) + old_re_min
        self._re_p2 = (self._rect_init_x + self._rect_dx)/ self.__width * (old_re_max - old_re_min) + old_re_min
        self._im_p1 = self._rect_init_y / self.__height * (old_im_max - old_im_min) + old_im_min
        self._im_p2 = (self._rect_init_y + self._rect_dy) / self.__height * (old_im_max - old_im_min) + old_im_min        
        self._re_min = min(self._re_p1, self._re_p2)
        self._re_max = max(self._re_p1, self._re_p2)
        self._im_min = min(self._im_p1, self._im_p2)
        self._im_max = max(self._im_p1, self._im_p2)
    
    @property
    def height(self):
        return self.__height
    
    @height.setter
    def height(self, value):
        self.__height = value
        self._adjust_coordinates_to_image_ratio()

    @property
    def width(self):
        return self.__width
    
    @width.setter
    def width(self, value):
        self.__width = value
        self._adjust_coordinates_to_image_ratio()

    def _adjust_coordinates_to_image_ratio(self):
        center_y = (self._im_max + self._im_min) / 2
        ratio_image = self.__width / self.__height
        dy = (self._re_max - self._re_min) / ratio_image / 2
        self._im_min = center_y - dy
        self._im_max = center_y + dy
        
if __name__ == '__main__':
    mb = Mandelbrot(800,1200,450)
    mb.run()
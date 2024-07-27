import threading
from time import perf_counter
from datetime import datetime
from pathlib import Path
import multiprocessing

import numpy as np
import cv2 as cv

def get_mandelbrot_iterations(c, z, max_iter):
    N = 10000
    P = 2
    log_N = np.log(N)
    for i in range(max_iter):
        z = z**P + c
        if abs(z) > N:
            nu = (i + 1) - np.log(np.log(abs(z)) / log_N) / np.log(P)
            return nu
    return float(max_iter)

def get_mandelbrot_iterations_image(params):
    col_coords, row_coords, z, max_iter = params
    # N = 10000
    # P = 2
    # log_N = np.log(N)
    iteration_image = np.zeros((len(row_coords), len(col_coords)), dtype=np.float16)
    for i, x in enumerate(col_coords):
        for j, y in enumerate(row_coords):
            c = complex(x, y)
            iterations = get_mandelbrot_iterations(c, z, max_iter)
            # for k in range(max_iter):
            #     z = z**P + c
            #     if abs(z) > N:
            #         iterations = (k + 1) - np.log(np.log(abs(z)) / log_N) / np.log(P)
            iteration_image[j,i] = iterations
    return iteration_image

class Mandelbrot:

    def __init__(self, height: int = 480, width: int = 640, max_iter: int = 50):
        self.single_core = False
        self.invert_in_set_color = False
        self.gamma = 1.0
        self.histogram_equalization_weight = 1.0

        self.__height = height
        self.__width = width
        self._init_coordinates = -2.0, 2.0, height / width * -2.0, height / width * 2.0
        self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
        self._max_iter = max_iter
        self._z = 0

        self._iteration_img = np.zeros((self.height, self.width), dtype=np.float16)
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._window_name = 'Mandelbrot Set'
        self._window = cv.namedWindow(self._window_name)        
        cv.setMouseCallback(self._window_name, self._mouse_callback)

        self._drawing_rectangle_event = threading.Event()
        self._calculate_mandelbrot_set_abort_flag = threading.Event()
        self._is_calculating_event = threading.Event()
        self._rect_init_x, self._rect_init_y = 0, 0
        self._rect_dx, self._rect_dy = 0, 0

        self._middle_mousebutton_states = [
            'change_max_iter',
            'change_resolution_width',
            'change_resolution_height',
            'change_gamma',
            'change_histogram_equalization_weight',
        ]
        self.middle_mousebutton_state = 0

    def run(self):
        self._calculate_mandelbrot_set()
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('i'):
                self.invert_in_set_color = not self.invert_in_set_color
                self._set_in_set_color()
            if key == ord('g'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_gamma')
                self._print_mousebutton_state()
            if key == ord('h'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_histogram_equalization_weight')
                self._print_mousebutton_state()
            if key == ord('r'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_resolution_width')
                self._print_mousebutton_state()
            if key == ord('t'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_resolution_height')
                self._print_mousebutton_state()
            if key == ord('m'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_max_iter')
                self._print_mousebutton_state()                
            if key == ord('s'):
                Path('images').mkdir(parents=True, exist_ok=True)
                date_and_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'images/mandelbrot_{date_and_time}.png'
                cv.imwrite(filename, self._image)
                print(f'Image saved: {filename}')
        cv.destroyAllWindows()

    def _repaint(self):
        cv.imshow(self._window_name, self._image)
        cv.waitKey(1)

    def _set_in_set_color(self):
        mask = self._iteration_img == self._max_iter
        val = 0 if self.invert_in_set_color else 255
        self._image[mask, :] = [val, val, val]

    @staticmethod
    def iterations_in_mandelbrot_set(c, z, max_iter):
        """
        Calculates the number of iterations required for a complex number `c` to escape the Mandelbrot set.

        Parameters:
        - c (complex): The complex number to be tested.
        - z (complex): The initial value of `z` in the iteration.
        - max_iter (int): The maximum number of iterations to perform.

        Returns:
        - float: The smoothed number of iterations required for `c` to escape the Mandelbrot set.

        """
        N = 10000
        P = 2
        log_N = np.log(N)
        for i in range(max_iter):
            z = z**P + c
            if abs(z) > N:
                nu = (i + 1) - np.log(np.log(abs(z)) / log_N) / np.log(P)
                return nu
        return float(max_iter)
    
    def _calculate_mandelbrot_set(self):
        self._is_calculating_event.set()
        self._calculate_mandelbrot_set_abort_flag.clear()
        self._image[:,:,:] = [0, 0, 0]
        self._iteration_img[:,:] = 0    
        global_t_0 = perf_counter()
        t_0 = perf_counter()
        row_coords = np.linspace(self._im_min, self._im_max, self.height)
        col_coords = np.linspace(self._re_min, self._re_max, self.width)
        if self.single_core:   
            for i, x in enumerate(col_coords):
                for j, y in enumerate(row_coords):
                    if self._calculate_mandelbrot_set_abort_flag.is_set():
                        self._is_calculating_event.clear()
                        return
                    c = complex(x, y)
                    iterations = self.iterations_in_mandelbrot_set(c, self._z, self._max_iter)
                    self._iteration_img[j, i] = iterations
                    preview_brightness = 0 if (self.invert_in_set_color and iterations == self._max_iter) else (iterations / self._max_iter * 255)
                    self._image[j, i, :] = [preview_brightness, preview_brightness, preview_brightness]
                if perf_counter() - t_0 > 0.1:
                    t_0 = perf_counter()
                    print(f'\r{" " * 50}\r{(x)/self.width*100:.2f}%', end='')
                    self._repaint()
        else:
            n_cores = multiprocessing.cpu_count()
            n_cols_per_core = self.width // n_cores
            col_coords_clipped = col_coords[:n_cores * n_cols_per_core]
            col_coords_split = [col_coords_clipped[i:i+n_cols_per_core] for i in range(0, len(col_coords_clipped), n_cols_per_core)]
            if len(col_coords) > n_cores * n_cols_per_core:
                col_coords_split[-1] = np.append(col_coords_split[-1], col_coords[n_cores * n_cols_per_core:])

            sum_of_lengths = sum(len(col_coords) for col_coords in col_coords_split)
            assert sum_of_lengths == len(col_coords), f'{sum_of_lengths} != {len(col_coords)}'

            with multiprocessing.Pool() as pool:
                params  = [(col_coords, row_coords, self._z, self._max_iter) for col_coords in col_coords_split]
                results = pool.map(get_mandelbrot_iterations_image, params)
            
            i = 0
            for col_coords, result in zip(col_coords_split, results):
                self._iteration_img[:, i:i+len(col_coords)] = result
                mask = result == self._max_iter
                brightness = (result / self._max_iter * 255).astype(np.uint8)
                brightness[mask] = 0 if self.invert_in_set_color else 255 
                i += len(col_coords)       
        self._colorize_img()        
        print()
        print(f'\r{" " * 50}\r100% ({perf_counter() - global_t_0:.2f}s)')
        self._print_mousebutton_state()
        self._is_calculating_event.clear()
        return
    
    def _colorize_img(self):
        min_iter = self._iteration_img.min()
        max_iter = self._iteration_img.max()
        normalized_smooth_values = (self._iteration_img - min_iter) / (max_iter - min_iter)
        
        # histogram_equalization'
        hist, bins = np.histogram(normalized_smooth_values, bins=256, density=True)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf[-1] - cdf.min())
        new_values = np.interp(normalized_smooth_values, bins[:-1], cdf)
        new_values = self.histogram_equalization_weight * new_values + (1 - self.histogram_equalization_weight) * normalized_smooth_values

        # gamma correction
        new_values = new_values ** (1 / self.gamma)

        self._image[:,:,:] = new_values[:,:,None] * 255
        if self.invert_in_set_color:
            self._set_in_set_color()
        self._repaint()
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
                self._repaint()
            else:
                self._calculate_mandelbrot_set_abort_flag.set() #TODO: problem: will not abort since initial function call gets paused
                self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
                self._calculate_mandelbrot_set()
                self._repaint()

        elif event == cv.EVENT_LBUTTONUP:
            if self._drawing_rectangle_event.is_set():
                self._drawing_rectangle_event.clear()
                if self._rect_dy != 0 and self._rect_dx != 0:
                    self._update_coordinates_from_rectangle()
                self._calculate_mandelbrot_set()
                self._repaint()

        elif event == cv.EVENT_MOUSEWHEEL:
            if self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_max_iter':
                delta = 20
                if flags > 0:
                    self._max_iter += delta
                else:
                    if self._max_iter <= delta:
                        self._max_iter = delta
                    else:
                        self._max_iter -= delta
                print(f'max_iter: {self._max_iter}')

            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_resolution_height':
                delta = 50
                if flags > 0:
                    self.height += delta # This will call the height setter and adjust coordinates accordingly
                else:
                    if self.height <= delta:
                        self.height = delta
                    else:
                        self.height -= delta # This will call the height setter and adjust coordinates accordingly
                print(f'height: {self.height}')
            
            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_resolution_width':
                delta = 50
                if flags > 0:
                    self.width += delta # This will call the width setter and adjust coordinates accordingly
                else:
                    if self.width <= delta:
                        self.width = delta
                    else:
                        self.width -= delta # This will call the width setter and adjust coordinates accordingly
                print(f'width: {self.width}')

            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_gamma':
                delta = 0.05
                if flags > 0:
                    self.gamma += delta
                else:
                    if self.gamma <= delta:
                        self.gamma = delta
                    else:
                        self.gamma -= delta
                self._colorize_img()
                print(f'gamma: {self.gamma}')
            
            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_histogram_equalization_weight':
                delta = 0.05
                if flags > 0:
                    if self.histogram_equalization_weight >= 1.0:
                        self.histogram_equalization_weight = 1.0
                    else:
                        self.histogram_equalization_weight += delta
                else:
                    if self.histogram_equalization_weight <= 0.0:
                        self.histogram_equalization_weight = 0.0
                    else:
                        self.histogram_equalization_weight -= delta
                self._colorize_img()
                print(f'histogram_equalization_weight: {self.histogram_equalization_weight}')

        elif event == cv.EVENT_MBUTTONUP:
            self.middle_mousebutton_state += 1
            self.middle_mousebutton_state %= len(self._middle_mousebutton_states)
            self._print_mousebutton_state()
    
    def _print_mousebutton_state(self):
        print(f'middle_mousebutton_state: {self._middle_mousebutton_states[self.middle_mousebutton_state]}')

    def _draw_rectangle(self, x, y):
        img_copy = self._image.copy()
        self._rect_dx = x - self._rect_init_x
        self._rect_dy = y - self._rect_init_y
        if self._rect_dx == 0 or self._rect_dy == 0:
            return
        # calculate width and height of rectangle according to image ratio
        ratio_image = abs(self.width / self.height)
        ratio_draw = abs(self._rect_dx / self._rect_dy)
        if ratio_draw < ratio_image:
            self._rect_dx = int(abs(self._rect_dy) * ratio_image) * np.sign(self._rect_dx)
        else:
            self._rect_dy = int(abs(self._rect_dx) / ratio_image) * np.sign(self._rect_dy)
        cv.rectangle(img_copy, (self._rect_init_x, self._rect_init_y), (self._rect_init_x + self._rect_dx, self._rect_init_y + self._rect_dy), (0, 0, 255), 1)
        cv.imshow(self._window_name, img_copy)

    def _update_coordinates_from_rectangle(self):
        old_re_min, old_re_max, old_im_min, old_im_max = self._re_min, self._re_max, self._im_min, self._im_max
        self._re_p1 = self._rect_init_x / self.width * (old_re_max - old_re_min) + old_re_min
        self._re_p2 = (self._rect_init_x + self._rect_dx)/ self.width * (old_re_max - old_re_min) + old_re_min
        self._im_p1 = self._rect_init_y / self.height * (old_im_max - old_im_min) + old_im_min
        self._im_p2 = (self._rect_init_y + self._rect_dy) / self.height * (old_im_max - old_im_min) + old_im_min        
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
        self._adjust_image_size()
        self._adjust_coordinates_to_image_ratio()

    @property
    def width(self):
        return self.__width
    
    @width.setter
    def width(self, value):
        self.__width = value
        self._adjust_image_size()
        self._adjust_coordinates_to_image_ratio()

    def _adjust_image_size(self):
        self._iteration_img = np.zeros((self.height, self.width), dtype=np.float16)
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _adjust_coordinates_to_image_ratio(self):
        center_y = (self._im_max + self._im_min) / 2
        ratio_image = self.width / self.height
        dy = (self._re_max - self._re_min) / ratio_image / 2
        self._im_min = center_y - dy
        self._im_max = center_y + dy
        
if __name__ == '__main__':
    mb = Mandelbrot(300,400,100)
    mb.gamma = 1.75
    mb.histogram_equalization_weight = 0.0
    mb.run()
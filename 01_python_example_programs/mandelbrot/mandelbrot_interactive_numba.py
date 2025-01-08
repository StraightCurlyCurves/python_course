import threading
from time import perf_counter
from datetime import datetime
from pathlib import Path
import threading

import numpy as np
import cv2 as cv
import numba
# import mpmath as mp

# ctx = mp.fp

@numba.jit(nopython=True, fastmath=True)
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

@numba.jit(nopython=True, fastmath=True, parallel=True)
def get_mandelbrot_iterations_image(col_coords, row_coords, z, max_iter):
    iteration_image = np.zeros((len(row_coords), len(col_coords)), dtype=np.float64)
    for i in numba.prange(len(col_coords)):
        x = col_coords[i]
        for j, y in enumerate(row_coords):
            c = complex(x, y)
            iterations = get_mandelbrot_iterations(c, z, max_iter)
            iteration_image[j,i] = iterations
    return iteration_image

@numba.jit(nopython=True, fastmath=True, parallel=True)
def get_buddhabrot_iterations_image(col_coords, row_coords, z, max_iter, num_points):
    iteration_image = np.zeros((len(row_coords), len(col_coords)), dtype=np.float64)
    for _ in numba.prange(num_points):
        random_x = np.random.uniform(col_coords[0], col_coords[-1])
        random_y = np.random.uniform(row_coords[0], col_coords[-1])
        c = complex(random_x, random_y)
        z = 0
        escapes = False
        for __ in range(max_iter):
            z = z**2 + c
            if abs(z) > 2:
                escapes = True
                break
        if escapes:
            z = 0
            for ___ in range(max_iter+1):
                z = z**2 + c
                if abs(z) > 2:
                    break
                x = np.interp(np.real(z), [col_coords[0], col_coords[-1]], [0, len(col_coords) - 1])
                y = np.interp(np.imag(z), [row_coords[0], col_coords[-1]], [0, len(row_coords) - 1])
                iteration_image[int(y), int(x)] += 1
    return iteration_image - iteration_image.min() / (iteration_image.max() - iteration_image.min())


class Mandelbrot:

    def __init__(self, height: int = 480, width: int = 640, max_iter: int = 100):
        self.invert_in_set_color = False
        self.gamma = 1.0
        self.histogram_equalization_weight = 1.0

        self.__height = height
        self.__width = width
        self._init_coordinates = -3.0, 3.0, height / width * -3.0, height / width * 3.0
        self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
        self.max_iter = max_iter
        self._z = 0

        self._iteration_image = np.zeros((self.height, self.width), dtype=np.float32)
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._window_name = 'Mandelbrot Set'
        self._window = cv.namedWindow(self._window_name)
        cv.setMouseCallback(self._window_name, self._mouse_callback)

        self._drawing_rectangle_zoom_in_event = False
        self._drawing_rectangle_zoom_out_event = False
        self._calculate_mandelbrot_set_abort_flag = False
        self._is_calculating_event = False
        self._rect_init_x, self._rect_init_y = 0, 0
        self._rect_dx, self._rect_dy = 0, 0

        self._history: list[list[float, float, float, float, np.ndarray]] = []        

        self._middle_mousebutton_states = [
            'change_max_iter',
            'change_resolution_width',
            'change_resolution_height',
            'change_gamma',
            'change_histogram_equalization_weight',
        ]
        self.middle_mousebutton_state = 0

    def run(self):
        self._repaint()
        self._print_help()
        self._calculate_mandelbrot_set()

        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break            
            if key == ord(' '):
                self._print_help()
            if key == ord('b'):
                self.invert_in_set_color = not self.invert_in_set_color
                self._set_in_set_color()
                self._repaint()
            if key == ord('r'):
                self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
                self._adjust_coordinates_to_image_ratio()
                self._history = False
                self._calculate_mandelbrot_set()
                print('Reset')
            if key == ord('g'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_gamma')
                self._print_mousebutton_state()
            if key == ord('h'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_histogram_equalization_weight')
                self._print_mousebutton_state()
            if key == ord('w'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_resolution_width')
                self._print_mousebutton_state()
            if key == ord('e'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_resolution_height')
                self._print_mousebutton_state()
            if key == ord('i'):
                self.middle_mousebutton_state = self._middle_mousebutton_states.index('change_max_iter')
                self._print_mousebutton_state()
            if key == ord('s'):
                Path('images').mkdir(parents=True, exist_ok=True)
                date_and_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                prefix = f'images/mandelbrot_{date_and_time}'
                image_filename = f'{prefix}.png'
                coordinates_filename = f'{prefix}_coordinates.txt'
                settings_filename = f'{prefix}_settings.txt'
                resolution_filename = f'{prefix}_resolution.txt'
                self.save_image(image_filename)
                self.save_coordinates(coordinates_filename)
                self.save_settings(settings_filename)
                self.save_resolution(resolution_filename)
                print(f'Image saved: {image_filename}')

            if key == ord('l'):
                filename = input('Enter filename: ')
                if filename.endswith('settings.txt'):
                    self.load_settings(filename)
                    self._colorize_image()
                    print(f'Loaded settings: {filename}')
                elif filename.endswith('coordinates.txt'):
                    self.load_coordinates(filename)
                    print(f'Loaded coordinates: {filename}')
                elif filename.endswith('resolution.txt'):
                    self.load_resolution(filename)
                    print(f'Loaded resolution: {filename}')
                else:
                    print('Invalid filename')

            if key == ord('p'):
                print(f're_min: {self._re_min}, re_max: {self._re_max}, im_min: {self._im_min}, im_max: {self._im_max}')
                print(f'height: {self.height}, width: {self.width}')
                print(f'max_iter: {self.max_iter}')
                print(f'gamma: {self.gamma}, histogram_equalization_weight: {self.histogram_equalization_weight}')
                self._print_mousebutton_state()

        cv.destroyAllWindows()

    def _print_help(self):
        print()
        print('Available commands:')
        print('space: show help')
        print('q: quit')
        print('p: print current position and settings')
        print('b: invert color of points in the mandelbrot set')
        print('r: reset coordinates to default')
        print('s: save image and current settings')
        print('l: load settings')
        print()
        print('Settings (scroll up/down to change the value):')
        print('g: change gamma')
        print('h: change histogram equalization weight')
        print('w: change resolution width')
        print('e: change resolution height')
        print('i: change maxximum number of iterations')
        print()
        self._print_mousebutton_state()
        print()

    def _repaint(self):
        cv.imshow(self._window_name, self._image)
        cv.waitKey(1)

    def _update_history(self):
        self._history.append([self._re_min, self._re_max, self._im_min, self._im_max, self._iteration_image.copy()])

    def _set_in_set_color(self):
        mask = self._iteration_image == self.max_iter
        val = 0 if self.invert_in_set_color else 255
        self._image[mask, :] = [val, val, val]
    
    def _calculate_mandelbrot_set(self):
        self._is_calculating_event = True
        self._calculate_mandelbrot_set_abort_flag = False
        self._image[:,:,:] = [0, 0, 0]
        self._iteration_image[:,:] = 0
        global_t_0 = perf_counter()
        row_coords = np.linspace(self._im_min, self._im_max, self.height)
        col_coords = np.linspace(self._re_min, self._re_max, self.width)
        if True:
            self._iteration_image = get_mandelbrot_iterations_image(col_coords, row_coords, self._z, self.max_iter)
            self._colorize_image()  
        else:            
            self._iteration_image = get_buddhabrot_iterations_image(col_coords, row_coords, self._z, self.max_iter, 1_000_000_000)
            # self._colorize_image()  
            self._image[:,:,:] = self._iteration_image[:,:,None]
        self._repaint() 
        print(f'\r{" " * 50}\r100% ({perf_counter() - global_t_0:.2f}s)')
        # print position
        print(f'pos: {(self._re_min+self._re_max)/2}, {(self._im_min+self._im_max)/2}')
        # print zoom level
        print(f'zoom level: {2/abs(self._re_max - self._re_min):.2e}')
        self._print_mousebutton_state()
        self._is_calculating_event = False
        return
    
    def _colorize_image(self):
        min_iter = self._iteration_image.min()
        max_iter = self._iteration_image.max()
        if min_iter == max_iter:
            self._image[:,:,:] = 255
        else:
            normalized_smooth_values = (self._iteration_image - min_iter) / (max_iter - min_iter)
            
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
            if not self._is_calculating_event:
                self._drawing_rectangle_zoom_in_event = True
                self._rect_init_x, self._rect_init_y = x, y
                self._rect_dx, self._rect_dy = 0, 0

        elif event == cv.EVENT_MOUSEMOVE:
            if self._drawing_rectangle_zoom_in_event:
                self._draw_rectangle(x, y)

        elif event == cv.EVENT_LBUTTONUP:
            if self._drawing_rectangle_zoom_in_event:
                self._drawing_rectangle_zoom_in_event = False
                if self._rect_dy != 0 and self._rect_dx != 0:
                    self._update_history()
                    self._update_coordinates_from_rectangle()
                    self._calculate_mandelbrot_set()
                else:
                    self._calculate_mandelbrot_set()
                self._repaint()

        elif event == cv.EVENT_RBUTTONUP:
            if self._drawing_rectangle_zoom_in_event:
                self._drawing_rectangle_zoom_in_event = False
                self._repaint()
            elif not self._is_calculating_event:
                if len(self._history) > 0:
                    self._re_min, self._re_max, self._im_min, self._im_max, _iteration_image = self._history[-1]
                    self.height, self.width = _iteration_image.shape # This will call the height and width setter and overwrite self._iteration_image
                    self._iteration_image = _iteration_image
                    self._colorize_image()
                    self._history.pop()

        elif event == cv.EVENT_MOUSEWHEEL:
            if self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_max_iter':
                delta = 20
                if flags > 0:
                    self.max_iter += delta
                else:
                    if self.max_iter <= delta:
                        self.max_iter = delta
                    else:
                        self.max_iter -= delta
                print(f'max_iter: {self.max_iter}')

            elif self._middle_mousebutton_states[self.middle_mousebutton_state] == 'change_resolution_height':
                delta = 50
                if flags > 0:
                    self.height += delta # This will call the height setter and adjust coordinates accordingly
                else:
                    if self.height <= delta:
                        self.height = delta
                    else:
                        self.height -= delta # This will call the height setter and adjust coordinates accordingly
                self._repaint()
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
                self._repaint()
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
                self._colorize_image()
                print(f'gamma: {self.gamma:.2f}')
            
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
                self._colorize_image()
                print(f'histogram_equalization_weight: {self.histogram_equalization_weight:.2f}')
    
    def _print_mousebutton_state(self):
        print(f'current setting: {self._middle_mousebutton_states[self.middle_mousebutton_state]}')

    def _draw_rectangle(self, x, y):
        image_copy = self._image.copy()
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
        cv.rectangle(image_copy, (self._rect_init_x, self._rect_init_y), (self._rect_init_x + self._rect_dx, self._rect_init_y + self._rect_dy), (0, 0, 255), 1)
        cv.imshow(self._window_name, image_copy)

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

    def save_image(self, filename: str):
        cv.imwrite(filename, self._image)
    
    def save_coordinates(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f're_min = {self._re_min}\n')
            f.write(f're_max = {self._re_max}\n')
            f.write(f'im_min = {self._im_min}\n')
            f.write(f'im_max = {self._im_max}\n')
        print(f'Coordinates saved: {filename}')

    def load_coordinates(self, filename: str):
        with open(filename, 'r') as f:
            self._re_min = float(f.readline().split('=')[1])
            self._re_max = float(f.readline().split('=')[1])
            self._im_min = float(f.readline().split('=')[1])
            self._im_max = float(f.readline().split('=')[1])
        self._adjust_coordinates_to_image_ratio()

    def save_settings(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f'max_iter = {self.max_iter}\n')
            f.write(f'invert_in_set_color = {self.invert_in_set_color}\n')
            f.write(f'gamma = {self.gamma}\n')
            f.write(f'histogram_equalization_weight = {self.histogram_equalization_weight}\n')

    def load_settings(self, filename: str):
        with open(filename, 'r') as f:
            self.max_iter = int(f.readline().split('=')[1])
            self.invert_in_set_color = bool(f.readline().split('=')[1])
            self.gamma = float(f.readline().split('=')[1])
            self.histogram_equalization_weight = float(f.readline().split('=')[1])

    def save_resolution(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f'width = {self.width}\n')
            f.write(f'height = {self.height}\n')

    def load_resolution(self, filename: str):
        with open(filename, 'r') as f:
            self.width = int(f.readline().split('=')[1])
            self.height = int(f.readline().split('=')[1])
        self._adjust_image_size()
        self._adjust_coordinates_to_image_ratio()

    def _adjust_image_size(self):
        self._iteration_image = np.zeros((self.height, self.width), dtype=np.float32)
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _adjust_coordinates_to_image_ratio(self):
        center_y = (self._im_max + self._im_min) / 2
        ratio_image = self.width / self.height
        dy = (self._re_max - self._re_min) / ratio_image / 2
        self._im_min = center_y - dy
        self._im_max = center_y + dy
    
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
        
if __name__ == '__main__':
    mb = Mandelbrot(800,800,1000)
    # mb._re_min = -0.74364386269 - 0.00000013526 / 2
    # mb._re_max = -0.74364386269 + 0.00000013526 / 2
    # mb._im_min = 0.13182590271 - 0.00000013526 / 2
    # mb._im_max = 0.13182590271 + 0.00000013526 / 2
    # mb._adjust_coordinates_to_image_ratio()
    mb.gamma = 1.75
    mb.histogram_equalization_weight = 0.0
    mb.run()
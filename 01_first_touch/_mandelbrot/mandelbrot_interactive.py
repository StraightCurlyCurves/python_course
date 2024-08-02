import threading
from time import perf_counter
from datetime import datetime
from pathlib import Path
import multiprocessing
from multiprocessing import sharedctypes
from multiprocessing import shared_memory
import threading

import numpy as np
import cv2 as cv

global_lock = multiprocessing.Lock()

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
    abort_flag, shared_array_base, shape, col_index, col_coords, row_coords, z, max_iter = params
    # col_coords, row_coords, z, max_iter = params
    global_image = np.ndarray(shape, dtype=np.uint8, buffer=shared_array_base.buf)
    abort_flag_np = np.ndarray(1, dtype=np.uint8, buffer=abort_flag.buf)
    iteration_image = np.zeros((len(row_coords), len(col_coords)), dtype=np.float16)
    t_0 = perf_counter()
    last_i = 0
    for i, x in enumerate(col_coords):
        for j, y in enumerate(row_coords):
            if abort_flag_np[0] == 1:
                return iteration_image
            c = complex(x, y)
            iterations = get_mandelbrot_iterations(c, z, max_iter)
            iteration_image[j,i] = iterations
        if perf_counter() - t_0 > 0.1:
            t_0 = perf_counter()
            last_i = i
            preview_brighntess = (iteration_image[:, :i] / max_iter * 255).astype(np.uint8)
            with global_lock:
                global_image[:, col_index:col_index + i] = preview_brighntess
    global_image[:, col_index+last_i:col_index + len(col_coords)] = (iteration_image[:, last_i:] / max_iter * 255).astype(np.uint8)
    return iteration_image

class Mandelbrot:

    def __init__(self, height: int = 480, width: int = 640, max_iter: int = 50):
        self.parallel_processing = True
        self._n_cores = multiprocessing.cpu_count()

        self.invert_in_set_color = False
        self.gamma = 1.0
        self.histogram_equalization_weight = 1.0

        self.__height = height
        self.__width = width
        self._init_coordinates = -2.0, 2.0, height / width * -2.0, height / width * 2.0
        self._re_min, self._re_max, self._im_min, self._im_max = self._init_coordinates
        self.max_iter = max_iter
        self._z = 0

        self._iteration_image = np.zeros((self.height, self.width), dtype=np.float32)
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._window_name = 'Mandelbrot Set'
        self._window = cv.namedWindow(self._window_name)
        cv.setMouseCallback(self._window_name, self._mouse_callback)

        self._drawing_rectangle_zoom_in_event = multiprocessing.Event()
        self._drawing_rectangle_zoom_out_event = multiprocessing.Event()
        self._calculate_mandelbrot_set_abort_flag = multiprocessing.Event()
        self._is_calculating_event = multiprocessing.Event()
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
                self._history.clear()
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
            if key == ord('m'):
                self.parallel_processing = not self.parallel_processing
                if self.parallel_processing:
                    print(f'Parallel Processing: active ({self._n_cores} cores)')
                else:
                    print('Parallel Processing: inactive (single core)')
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
        print('m: toggle parallel / single core processing')
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

    @staticmethod
    def iterations_in_mandelbrot_set(c, z, max_iter):
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
        self._iteration_image[:,:] = 0
        global_t_0 = perf_counter()
        t_0 = perf_counter()
        row_coords = np.linspace(self._im_min, self._im_max, self.height)
        col_coords = np.linspace(self._re_min, self._re_max, self.width)
        if not self.parallel_processing:
            for i, x in enumerate(col_coords):
                for j, y in enumerate(row_coords):     
                    if self._calculate_mandelbrot_set_abort_flag.is_set():
                        self._is_calculating_event.clear()
                        print()
                        return
                    c = complex(x, y)
                    iterations = self.iterations_in_mandelbrot_set(c, self._z, self.max_iter)
                    self._iteration_image[j, i] = iterations
                    preview_brightness = 0 if (self.invert_in_set_color and iterations == self.max_iter) else (iterations / self.max_iter * 255)
                    self._image[j, i, :] = [preview_brightness, preview_brightness, preview_brightness]
                if perf_counter() - t_0 > 0.1:
                    t_0 = perf_counter()
                    print(f'\r{" " * 50}\r{(i)/self.width*100:.2f}%', end='')
                    self._repaint()
        else:
            n_cols_per_core = self.width // self._n_cores
            col_coords_clipped = col_coords[:self._n_cores * n_cols_per_core]
            col_coords_split = [col_coords_clipped[i:i+n_cols_per_core] for i in range(0, len(col_coords_clipped), n_cols_per_core)]
            image_indices = [i for i in range(0, len(col_coords_clipped), n_cols_per_core)]
            if len(col_coords) > self._n_cores * n_cols_per_core:
                col_coords_split[-1] = np.append(col_coords_split[-1], col_coords[self._n_cores * n_cols_per_core:])
                image_indices.append(self._n_cores * n_cols_per_core)

            sum_of_lengths = sum(len(col_coords) for col_coords in col_coords_split)
            assert sum_of_lengths == len(col_coords), f'{sum_of_lengths} != {len(col_coords)}'
            
            shared_array_base =  shared_memory.SharedMemory(create=True, size=self.height * self.width)
            shared_array_np = np.ndarray((self.height, self.width), dtype=np.uint8, buffer=shared_array_base.buf)
            abort_flag = shared_memory.SharedMemory(create=True, size=1)
            abort_flag_np = np.ndarray(1, dtype=np.uint8, buffer=abort_flag.buf)
            abort_flag_np[0] = False
            def multiprocess_image_calculation():
                with multiprocessing.Pool(self._n_cores) as pool:
                    params  = [(abort_flag, shared_array_base, (self.height, self.width), i, col_coords, row_coords, self._z, self.max_iter) for (i, col_coords) in zip(image_indices, col_coords_split)]
                    results = pool.map(get_mandelbrot_iterations_image, params)
                i = 0
                for col_coords, result in zip(col_coords_split, results):
                    self._iteration_image[:, i:i+len(col_coords)] = result
                    mask = result == self.max_iter
                    brightness = (result / self.max_iter * 255).astype(np.uint8)
                    brightness[mask] = 0 if self.invert_in_set_color else 255 
                    i += len(col_coords)
            thread = threading.Thread(target=multiprocess_image_calculation)
            thread.start()
            while thread.is_alive():
                cv.imshow(self._window_name, shared_array_np)
                key = cv.waitKey(1)
                if key == ord('c') or self._calculate_mandelbrot_set_abort_flag.is_set():
                    abort_flag_np[0] = 1
                    self._is_calculating_event.clear()
                    thread.join()
                    break           
        self._colorize_image()
        print(f'\r{" " * 50}\r100% ({perf_counter() - global_t_0:.2f}s)')
        self._print_mousebutton_state()
        self._is_calculating_event.clear()
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
            self._calculate_mandelbrot_set_abort_flag.set()
            self._drawing_rectangle_zoom_in_event.set()
            self._rect_init_x, self._rect_init_y = x, y
            self._rect_dx, self._rect_dy = 0, 0

        if event == cv.EVENT_RBUTTONDOWN:
            self._calculate_mandelbrot_set_abort_flag.set()

        elif event == cv.EVENT_MOUSEMOVE:
            if self._drawing_rectangle_zoom_in_event.is_set():
                self._draw_rectangle(x, y)

        elif event == cv.EVENT_RBUTTONUP:
            if self._drawing_rectangle_zoom_in_event.is_set():
                self._drawing_rectangle_zoom_in_event.clear()
                self._repaint()
            else:
                while self._is_calculating_event.is_set():
                    pass
                if len(self._history) > 0:
                    self._re_min, self._re_max, self._im_min, self._im_max, _iteration_image = self._history[-1]
                    self.height, self.width = _iteration_image.shape # This will call the height and width setter and overwrite self._iteration_image
                    self._iteration_image = _iteration_image
                    self._colorize_image()
                    self._history.pop()

        elif event == cv.EVENT_LBUTTONUP:
            if self._drawing_rectangle_zoom_in_event.is_set():
                self._drawing_rectangle_zoom_in_event.clear()
                if self._rect_dy != 0 and self._rect_dx != 0:
                    self._update_history()
                    self._update_coordinates_from_rectangle()
                    self._calculate_mandelbrot_set()
                else:
                    self._calculate_mandelbrot_set()
                self._repaint()

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

        elif event == cv.EVENT_MBUTTONUP:
            self.middle_mousebutton_state += 1
            self.middle_mousebutton_state %= len(self._middle_mousebutton_states)
            self._print_mousebutton_state()
    
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
    mb = Mandelbrot(400,400,100)
    mb._re_min = -0.74364386269 - 0.00000013526 / 2
    mb._re_max = -0.74364386269 + 0.00000013526 / 2
    mb._im_min = 0.13182590271 - 0.00000013526 / 2
    mb._im_max = 0.13182590271 + 0.00000013526 / 2
    mb._adjust_coordinates_to_image_ratio()
    mb.gamma = 1.75
    mb.histogram_equalization_weight = 0.0
    mb.run()
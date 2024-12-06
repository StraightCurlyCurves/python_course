import numpy as np
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_sine_functions():
    x_data = np.linspace(0, 2, 1000)

    def f_orig(x, freq, freq_shift, phase_shift):
        ax.set_title(f'Frequency: {freq + freq_shift:.2f} Hz, Phase: {phase_shift * 180 / np.pi:.2f}°')
        return 0.5 * np.sin(2 * np.pi * freq * x)

    def f_mod(x, freq, freq_shift, phase_shift):
        return 0.5 * np.sin(2 * np.pi * (freq + freq_shift) * x + phase_shift)

    def f_sum(x, freq, freq_shift, phase_shift):
        return f_orig(x, freq, freq_shift, phase_shift) + f_mod(x, freq, freq_shift, phase_shift)

    fig, ax = plt.subplots()

    # Create interactive plots
    controls = iplt.plot(x_data, f_orig, freq=(1, 10, 101), freq_shift=(0, 5, 101), phase_shift=(0, 2*np.pi, 101), label='original')
    iplt.plot(x_data, f_mod, controls=controls, label='modified')
    iplt.plot(x_data, f_sum, controls=controls, label='sum')

    # Add legend and labels
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-1, 1)

    plt.show()

def mouse_callback():
    # Define a 2D function
    def waves(x, y):
        return np.sin(x) * np.cos(3*y)
    
    # Create 2D data
    x_max, y_max = 10, 5
    x_data = np.linspace(0, x_max, 1000)
    y_data = np.linspace(0, y_max, 500)
    X, Y = np.meshgrid(x_data, y_data)
    Z = waves(X, Y) # 2D image

    # Create two figures with one axis each
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Dynamic content in ax1
    x_line = Line2D([], [], color='tab:blue')
    y_line = Line2D([], [], color='tab:red')
    ax1.add_line(x_line)
    ax1.add_line(y_line)

    # Dynamic content in ax2
    X_line, = ax2.plot([], [], label='X: f(x,y), fixed y', color='tab:blue')
    Y_line, = ax2.plot([], [], label='Y: f(x,y), fixed x', color='tab:red')
    Y_intersec_line, = ax2.plot([], [], 'o', label='Y-intersection', color='tab:red')
    X_intersec_line, = ax2.plot([], [], 'o', label='X-intersection', color='tab:blue')    

    # Mouse callback function
    def on_hover(event):
        if event.inaxes != ax1:
            return
        
        x, y = event.xdata, event.ydata

        # Update the lines in ax1
        x_line.set_data([0, x_max], [y, y])
        y_line.set_data([x, x], [0, y_max])            
        fig1.canvas.draw_idle()

        # Update the content in ax2
        y_data_horizontal = waves(x_data, y)
        y_data_vertical = waves(x, y_data)
        z = waves(x, y)
        X_line.set_data(x_data, y_data_horizontal)
        X_line.set_label(f'X: f(x,y), y={y:.2f}')
        Y_line.set_data(y_data, y_data_vertical)
        Y_line.set_label(f'Y: f(x,y), x={x:.2f}')
        Y_intersec_line.set_data([x], [z])
        X_intersec_line.set_data([y], [z])
        ax2.legend()          
        fig2.canvas.draw_idle()

    # Connect the callback function to the heatmap figure
    fig1.canvas.mpl_connect('motion_notify_event', on_hover)

    # Plot the heatmap in ax1 and set properties
    heatmap = ax1.imshow(Z, extent=[0, x_max, 0, y_max], origin='lower', cmap='viridis')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(heatmap, ax=ax1, cax=cax)
    ax1.set_title('Heatmap of f(x,y) = sin(x) * cos(3y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Set properties for ax2
    ax2.set_title('Sine Wave Plot')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)
    ax2.grid(True)
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    plot_sine_functions()
    mouse_callback()
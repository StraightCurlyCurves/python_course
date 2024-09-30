# Install Python and IDE (Integrated Development Environment)

To start with Python, the initial step is installing the Python interpreter. This enables the execution of Python (code) directly in the terminal or through scripts created with any text editor, including those provided by the operating system.

However, for an enhanced development experience, the utilization of an Integrated Development Environment (IDE) is highly recommended. IDEs offer a suite of features designed to facilitate programming tasks, such as syntax highlighting, code completion, and real-time error detection, among others.

So let's install Python first, test it quickly in the terminal and then install Visual Studio Code, one of the most used IDE.

## Install Python

There are various approaches to installing Python, each catering to different preferences and requirements. One can download Python directly from its official website, python.org, which is often recommended for beginners due to its simplicity and direct access to the latest, stable version. Alternatively, package managers like Conda simplify managing package dependencies, especially in data science, while distributions like WinPython provide a portable, pre-configured setup ideal for quick starts on Windows.

For this course, we will install Python directly from python.org. It's lightweight and straightforward, allowing us to install only the packages we need, avoiding unnecessary complexity. This approach provides a clean, efficient setup ideal for beginners and experienced programmers alike.

### For Windows

1. Go to [python.org](https://www.python.org) and download the latest Python interpreter (hover over "Downloads", click on the recommended version).
2. Execute the downloaded file.
3. Check the box "Add Python to PATH". This allows the use of the `py` or `python` command in the terminal to invoke the Python interpreter without needing to specify the full path to the `python.exe` file.
4. Click install.
5. Check "Disable maximum path length". This allows Python to handle file paths exceeding the default 260-character limit in Windows, preventing potential path-related errors.
6. Finish the installation process.

Python should now be successfully installed and ready to be used:
- Open up a terminal (in windows search, search for "cmd" and click on "Command Prompt").
- Type "py" and hit enter. You are now in the Python interpreter's interactive mode, ready to execute Python commands directly from the terminal.

- To execute a simple addition operation and display the result in the terminal, follow these steps:

  1. Assign the value `2` to variable `a`:
      ```python
      a = 2
      ```
  2. Assign the value `3` to variable `b`:
      ```python
      b = 3
      ```
  3. Calculate the sum of `a` and `b`, and assign it to variable `c`:
      ```python
      c = a + b
      ```
  4. Display the result by printing `c`:
      ```python
      print(c)
      ```

    After completing these steps, the terminal should output the number 5.

You can also place these four lines of code into a Python file, essentially a text file that uses the `.py` extension instead of `.txt`. This file can be edited with any text editor. To execute the code, you would then run this file using the following command.

```
py path/to/python_file.py
```



### For Linux

You will figure it out...

### For MacOS



## Install and setup Visual Studio Code

Installing Visual Studio Code (VS Code) is fairly simple: Go to [https://code.visualstudio.com/](https://code.visualstudio.com/) and Download the suggested version and install it.

In VS Code it is possible to install `Extensions`, which help us in several ways. Let's install some of the handy ones: Launch VS Code and navigate to the Extensions view by clicking on the Extensions icon located in the sidebar on the left (or by the shortcut Ctrl+Shift+X). Search for and install following extensions:
- `Python`: Adds several functionality to VS Code for Python programming, for example:
  - A "Run" button for executing an open Python script directly, bypassing terminal commands.
  - Ability to select a specific Python version (which is installed on your system) or virtual environment to run a script.
  - ...
- `Pylance`: Will be installed automatically with the Python extension. Adds functionality such as:
  - Enhanced syntax highlighting for improved code readability.
  - Advanced code autocompletion to speed up development.
  - ...
- `Jupyter`: Integrates Jupyter notebook support within VS Code, offering capabilities such as:
  - Partial execution of code blocks, allowing for iterative testing and development.\
  Best is to try it out (this course contains quite some of them), then you see what they're good for. 

You are now ready to go!

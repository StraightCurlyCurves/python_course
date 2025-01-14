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


### Manage multiple Python versions with the Python Launcher

For Linux / macOS, the python launcher has to be installed seperately. Follow the instructions here: https://python-launcher.app/install/

The `py` command doesn't directly call `python.exe`. Instead, it invokes the Python Launcher, which is included by default when installing Python from [python.org](https://www.python.org) on Windows. The Python Launcher tracks the different Python versions installed on the system and links the `py` command to a default version of Python.

To see the installed versions and the default one (*), type:

```
py -0p
```
Example output:
```
-V:3.12 *        C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe
-V:3.11          C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe
-V:3.9           C:\Users\<username>\AppData\Local\Programs\Python\Python39\python.exe
```

To start a specific Python version or execute a script with a specific version, type:

```
py -3.xx path/to/python_file.py
```

To change the default Python version, type the following in a terminal with admin rights (open terminal as administrator):

```
setx PY_PYTHON 3.xx /M
```

Restart the terminal and check with `py -0` if the default has changed. If not, there might be a `py.ini` file in the launcher folder or in your user's home directory, and you will need to change the version in there manually, as this file has a higher precedence than the system variable.

### Pip with mutliple Python versions

When you use `pip` to install packages (see chapter `pip` in `modules`), it always uses the `pip.exe` found first in the system's PATH environment variable. This means that packages are installed only for the Python version associated with that `pip.exe`, regardless of the interpreter selected in Visual Studio Code (which only affects the execution of Python files). However, if you have a virtual environment activated in the terminal, pip will install packages for the Python version of that virtual environment, but that is a different topic.

To install a package for a specific Python version, type:

```
py -3.xx -m pip install <package_name>
```

To change the default pip version, modify the system's PATH environment variables by moving the Python Scripts directory of the desired Python version to the top of the list. This ensures that the corresponding pip.exe is found first when executing pip commands.

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

# Virtual environments

Let us say you are working on two Python projects, Project A and Project B. For compatibility with other packages needed in Project A, you need to use NumPy 1.24.1. On the other hand, in Project B, you want all the latest features of NumPy, so you want to use the version 2.1.1 (which is the latest version at the time of writing this). 
Installing both NumPy 1.24.1 and 2.1.1 globally might cause conflicts.

The recommended way to manage this situation is to use virtual environments.


## Introduction - What is a virtual environment?

A virtual environment is an isolated workspace that allows you to manage dependencies for a specific Python project without affecting the system-wide Python installation or other projects.

To use a virtual environment, you need to:
1. Create a virtual environment;
2. Activate the virtual environment;
3. Install the desired packages.

In the rest of this lesson, we will go over these steps in more detail.


## Working with virtual environments

### Creating a virtual environment

Before trying to create a virtual environment in VS Code, make sure that you are inside the workspace of your project (i.e. make sure you go to File > Open Folder and select your project, instead of just opening a single file in VS Code).

To create a virtual environment:
1. Open the Command Palette, search for the "Python: Create Environment" command, and select it;
2. Select "Venv";
3. Choose the Python interpreter to be used as a base for your virtual environment.


### Activating a virtual environment

If you followed the steps above, your newly created virtual environment should automatically be activated. 

You have successfully activated the virtual environment if in the bottom right corner you see not only the Python version, but also the name of your virtual environment.
Alternatively, if you go to Terminal > New Terminal and in the terminal window you see the name of your virtual environment in brackets at the start of the line (for example, `(.venv) user project %`), the virtual environment is active. 

If the virtual environment is not active (or if you wish to select a different virtual environment), go to Command Palette, search for "Python: Select Interpreter" command, select it, and then select the interpreter you want.

If the virtual environment is not in the list that appears after selecting "Python: Select Interpreter" and if you know where the virtual environment is located on your computer, select the "Enter interpreter path..." option and browse for the interpreter on your file system or provide the path to it manually.


### Installing packages into a virtual environment

Once the virtual environment is active, you can install packages as you normally would, and they will be installed into the virtual environment.


## Tip

It is good practice to have a separate virtual environment for each project you work on.

Keeping dependencies isolated in this way has another benefit: it is easier to keep track of what you are actually using in your project. This is particularly important if you want to create a list of requirements for your software (these make it easy for other users to install the dependencies needed to be able to use your project).

A virtual environment is not something you would normally put into version control or include with your software. 
If you're keeping your projects in the cloud, we recommend against syncing the virtual environments, as this may cause conflicts.


## Summary

In this lesson we introduced the concept of virtual environments, isolated workspaces that allow dependency management without affecting the system-wide Python installation or other projects.

We learned how to create virtual environments, how to activate them, and how to use them in VS Code.

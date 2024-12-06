## Introduction

In this section, we will explore the concept of modules in Python and the importance of the if __name__ == "__main__" construct. Understanding these concepts is crucial for writing modular and reusable code, as well as for controlling the execution of scripts.

## Problem

When writing Python scripts, it's common to create reusable functions and classes that can be imported into other scripts. However, if the script contains code that should only run when the script is executed directly, such as test code or example usage, this code will also run when the script is imported as a module. This can lead to unintended side effects, such as unwanted print statements, execution of test code, or other operations that should not occur during an import.

## Solution

To prevent this, we use the if `__name__ == "__main__"` construct. This ensures that certain parts of the code are only executed when the script is run directly, and not when it is imported as a module. Without this construct, any code outside of function and class definitions will run upon import, potentially causing issues in the importing script.

## Import a Module, Class, or Function

To import code from one module into another, you can use the import statement:

```python
# Import the entire module
import my_module

# Use a function from the module
result = my_module.my_function()
```

```python
# Import a specific class from the module
from my_module import MyClass

# Create an instance of the imported class
instance = MyClass()
```

Run `main.py` and see what happens with code with / without the `__name__ == "__main__"` construct.
import module_A
import module_B
from other_modules.module_C import fun_C
from other_modules.future.module_D import fun_D

if __name__ == "__main__":
    module_A.fun_A("test string from main")
    module_B.fun_B("test string from main")
    fun_C("test string from main")
    fun_D("test string from main")
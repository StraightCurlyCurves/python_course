def fun_A(s):
    print(f"module A function test print: {s}")

# test function with __name__ == "__main__"
if __name__ == "__main__":
    fun_A("test string from Module A")
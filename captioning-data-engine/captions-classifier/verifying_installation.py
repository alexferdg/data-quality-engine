import triton.language as tl

def check_function():
    if hasattr(tl.math, 'tanh'):
        print("tanh function is available.")
    else:
        print("tanh function is NOT available.")

check_function()
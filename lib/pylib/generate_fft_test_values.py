import numpy as np
import numpy.fft as fft

PI_STR = "PI"

def print_mag(arr):
    arr = np.abs(arr)
    arr = [f"{num:.8g}" for num in arr]
    arr = ", ".join(arr)
    print(f"mag: [{arr}]")

def print_phase(arr):
    arr = np.angle(arr) / np.pi
    arr = [f"{num:.8g}*{PI_STR}" for num in arr]
    arr = ", ".join(arr)
    print(f"phase: [{arr}]")

def print_input(arr):
    arr = [f"{num:.8g}" for num in arr]
    arr = ", ".join(arr)
    print(f"input: [{arr}]")

###############################################################
# 8-point FFT test cases
###############################################################
print("DiscreteFourierTest.BasicTransforms")

# case 1
print("Test Case 1")
input = np.ones(8)
output = fft.fft(input)
print_mag(output)
print_phase(output)

# case 2
print("Test Case 2")
input = np.zeros(8)
input[2] = 1
output = fft.fft(input)
print_mag(output)
print_phase(output)

# case 3
print("Test Case 3")
input = np.array([0, 6, -1, 3, 3, 0, -5, 2])
output = fft.fft(input)
print_mag(output)
print_phase(output)

###############################################################
# Larger FFT test cases
###############################################################
print("DiscreteFourierTest.LargeTransforms")

rng = np.random.default_rng(seed=1170)

# case 1
print("Test Case 1")
input = rng.random(16)
output = fft.fft(input)
print_input(input)
print_mag(output)
print_phase(output)

###############################################################
# Identity tests: input -> FFT -> IFFT -> input 
###############################################################
print("DiscreteFourierTest.IdentityTransforms")

# case 1
print("Test Case 1")
input = -1 + 2*rng.random(8)
print_input(input)

# case 2
print("Test Case 2")
input = -5 + 10*rng.random(16)
print_input(input)


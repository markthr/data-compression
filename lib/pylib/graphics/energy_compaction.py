import numpy as np
import matplotlib.pyplot as plt
import transforms as tm

size = 16
energy_thresh = 0.9
fft_16 = tm.fft(size)
dct_16 = tm.dct_2(size)

names = ["FFT", "DCT2"]

transformers = [fft_16, dct_16]

impulse = np.zeros(size)
impulse[14] = 1

results = [tr.transform(impulse) for tr in transformers]

energy = [[np.real(np.vdot(coeff, coeff)) for coeff in row] for row in results]
sums = [np.sum(row) for row in energy]
energy = np.array(energy).transpose()
energy = energy/sums

fig, ax = plt.subplots()
ax.semilogy(energy, label=names)
ax.set_title("Energy distribution of an impulse")
ax.set_xlabel("Frequency")
ax.set_ylabel("Energy")
ax.legend()

psums = [0, 0]
cutoffs = [size, size]

# simple compaction scheme, remove high frequencies while staying below a threshold of energy loss
for j, col in enumerate(energy.T):
    for val in reversed(col):
        if (psums[j] + val < 1 - energy_thresh):
            psums[j] += val
            cutoffs[j] -= 1
        else:
            break

compacted = np.array(results).transpose()
compacted[cutoffs[0]:, 0] = 0
compacted[cutoffs[1]:, 1] = 0

output = np.empty(compacted.shape)
output[:, 0] = transformers[0].inverse(compacted[:, 0])
output[:, 1] = transformers[1].inverse(np.real(compacted[:, 1]))

fig, ax = plt.subplots()
cmp_names = names.copy()
cmp_names[0] += f" (encoded with {(cutoffs[0] -1) *2} terms)"
cmp_names[1] += f" (encoded with {(cutoffs[1] -1)} terms)"
ax.plot(output, label=cmp_names)
ax.set_title("Reconstruction of an impulse")
ax.set_xlabel("Sample")
ax.set_ylabel("Value")
ax.legend()

plt.show()
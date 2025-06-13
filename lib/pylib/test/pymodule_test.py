import numpy as np
import transforms as tm
dct = tm.dct_2(8)
res = dct.transform(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
print(res)
inv_res = dct.inverse(res)
print(inv_res)

fft = tm.fft(8)
res = fft.transform(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
print(res)
inv_res = fft.inverse(res)
print(inv_res)
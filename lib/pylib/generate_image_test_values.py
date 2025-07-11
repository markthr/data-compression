from PIL import Image
import numpy as np

import os

print(os.getcwd())
# image taken from https://commons.wikimedia.org/wiki/File:Orioles_Mascot.jpg
pic = Image.open("img/Orioles_Mascot_Cropped.jpg")

pix = np.array(pic)
pix_f = np.reshape(pix, (-1, pix.shape[1]*3))

np.savetxt("data/orioles_mascot_cropped.csv", pix_f.astype(int), delimiter=',', fmt="%i")
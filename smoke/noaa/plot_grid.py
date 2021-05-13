import matplotlib.pyplot as plt
import numpy as np


grid2D = np.load(r"C:\temp\10km_grids\20180808-23.npy")

fig, ax = plt.subplots(figsize=(16.2, 16))
im = ax.imshow(grid2D)
ax.set_xlabel("Cols")
ax.set_ylabel("Rows")
plt.colorbar(im)

plt.savefig('grid2D.png')

import numpy as np
import matplotlib.pyplot as plt

y = np.load("loss_one_image.npy")

x = np.arange(1,len(y)+1)

plt.plot(x, y, label="batch=3, origin lr, one image")
plt.ylabel("loss value", fontsize=16)
plt.xlabel('updates', fontsize=16)

plt.legend()
plt.tight_layout()
plt.show()


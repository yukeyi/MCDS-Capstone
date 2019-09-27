import numpy as np
import matplotlib.pyplot as plt

y = np.load("loss_different_images.npy")

x = np.arange(1,len(y)+1)

plt.plot(x, y, label="batch=3")
plt.ylabel("loss value", fontsize=16)
plt.xlabel('updates', fontsize=16)

plt.legend()
plt.tight_layout()
plt.show()


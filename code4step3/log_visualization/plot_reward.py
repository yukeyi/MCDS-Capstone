import matplotlib.pyplot as plt
import numpy as np

loss = np.load("6layer_loss.npy")
loss = loss[100:]
x = np.arange(len(loss))

plt.plot(x, loss, label="6 layer conv")
plt.ylabel("Loss value", fontsize=16)
#plt.ylabel("Mean Absolute Error", fontsize=16)
plt.xlabel("Number of Images", fontsize=16)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
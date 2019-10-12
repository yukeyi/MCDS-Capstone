import matplotlib.pyplot as plt
import numpy as np

loss = np.load("loss_different_images_large_model2.npy")
loss = loss[2:]
x = np.arange(len(loss))

plt.plot(x, loss)
plt.ylabel("Loss value", fontsize=16)
#plt.ylabel("Mean Absolute Error", fontsize=16)
plt.xlabel("Number of Images", fontsize=16)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
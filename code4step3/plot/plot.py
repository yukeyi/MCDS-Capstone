import matplotlib.pyplot as plt
import numpy as np

loss = np.load("loss.npy")
pos = np.load("distance.npy")[0]
pos_mean = pos.reshape(-1,2).T[0]
neg = np.load("distance.npy")[1]
neg_mean = neg.reshape(-1,2).T[0]
x = np.arange(len(loss))

plt.plot(x, loss, label="test model")
#plt.plot(x, pos_mean, label="pos")
#plt.plot(x, neg_mean, label="neg")
plt.ylabel("distance", fontsize=16)
#plt.ylabel("Mean Absolute Error", fontsize=16)
plt.xlabel("Number of Images", fontsize=16)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
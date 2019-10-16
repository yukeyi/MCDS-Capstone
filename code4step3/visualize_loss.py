import numpy as np
import matplotlib.pyplot as plt
import os

#print(os.listdir("."))
y2 = np.load("loss1.npy")
y2 = y2[10:]
#y1 = np.load("loss_1channel1.npy")

#x1 = np.arange(1,len(y1)+1)
x2 = np.arange(1,len(y2)+1)

plt.plot(x2, y2, label="differnet image, 10000 points, large model")
#plt.plot(x1, y1, label="one image, 10000 points, out channel = 1")
plt.ylabel("loss value", fontsize=16)
plt.xlabel('images', fontsize=16)

plt.legend()
plt.tight_layout()
plt.show()


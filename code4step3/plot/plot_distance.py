import matplotlib.pyplot as plt
import numpy as np

distance0 = np.load("distance_easy.npy")
pos_mean0 = distance0[0].reshape((-1,2)).transpose()[0]
neg_mean0 = distance0[1].reshape((-1,2)).transpose()[0]

distance1 = np.load("distance.npy")
pos_mean1 = distance1[0].reshape((-1,2)).transpose()[0]
neg_mean1 = distance1[1].reshape((-1,2)).transpose()[0]

pos_mean0 = np.array(list(pos_mean0)+list(pos_mean1))
neg_mean0 = np.array(list(neg_mean0)+list(neg_mean1))


'''
distance1 = np.load("distance1.npy")
pos_mean1 = distance1[0].reshape((-1,2)).transpose()[0]
neg_mean1 = distance1[1].reshape((-1,2)).transpose()[0]

distance2 = np.load("distance2.npy")
pos_mean2 = distance2[0].reshape((-1,2)).transpose()[0]
neg_mean2 = distance2[1].reshape((-1,2)).transpose()[0]

distance3 = np.load("distance3.npy")
pos_mean3 = distance3[0].reshape((-1,2)).transpose()[0]
neg_mean3 = distance3[1].reshape((-1,2)).transpose()[0]

pos_mean = np.concatenate((pos_mean0,pos_mean1,pos_mean2,pos_mean3))
neg_mean = np.concatenate((neg_mean0,neg_mean1,neg_mean2,neg_mean3))
'''
print(len(pos_mean0))
x1 = np.arange(len(pos_mean0))
x2 = np.arange(len(neg_mean0))

plt.plot(x1, pos_mean0, label="positive")
plt.plot(x2, neg_mean0, label="negative")

plt.vlines(0, 0.2, 2.8, colors = "gray", linestyles = "dashed")
plt.vlines(625, 0.2, 2.8, colors = "gray", linestyles = "dashed")
plt.vlines(1250, 0.2, 2.8, colors = "gray", linestyles = "dashed")
plt.vlines(2520, 0.2, 2.8, colors = "gray", linestyles = "dashed")
plt.vlines(3110, 0.2, 2.8, colors = "gray", linestyles = "dashed")

#plt.hlines(0.5, 1, len(x2), colors = "gray", linestyles = "dashed")

plt.ylabel("Distance value", fontsize=16)
#plt.ylabel("Mean Absolute Error", fontsize=16)
plt.xlabel("Number of Images, Feature size = 32", fontsize=16)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

'''
sample_dev_acc = np.load("sample_data/dev_acc.npy")[:-1]
sample_dev_dice = np.load("sample_data/dev_dice.npy")[:-1]

x1 = np.arange(len(sample_dev_acc))
x2 = np.arange(len(sample_dev_dice))


plt.plot(x1, sample_dev_acc, label="Balanced data (70K)")
#plt.plot(x2, sample_dev_dice, label="Balanced data (70K)")


#plt.hlines(1.8, 1, len(x2), colors = "gray", linestyles = "dashed")
#plt.hlines(0.5, 1, len(x2), colors = "gray", linestyles = "dashed")

plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Number of epochs", fontsize=16)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
'''

full_dev_acc = list(np.load("full_data_1/dev_acc.npy"))+list(np.load("full_data_2/dev_acc.npy"))+[82.15]
full_dev_dice = list(np.load("full_data_1/dev_dice.npy"))+list(np.load("full_data_2/dev_dice.npy"))+[0.342]

x1 = np.arange(len(full_dev_acc))
x2 = np.arange(len(full_dev_dice))


#plt.plot(x1, full_dev_acc, label="Original data (3.4M)")
plt.plot(x2, full_dev_dice, label="Original data (3.4M)")


#plt.hlines(1.8, 1, len(x2), colors = "gray", linestyles = "dashed")
#plt.hlines(0.5, 1, len(x2), colors = "gray", linestyles = "dashed")

plt.ylabel("Dice Score", fontsize=16)
plt.xlabel("Number of epochs", fontsize=16)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
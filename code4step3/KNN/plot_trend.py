import matplotlib.pyplot as plt
import numpy as np

model8y = [5.8,6.0,6.2]
model8x = [0.5,1,2]

model16y = [5.2,5.5,5.5]
model16x = [0.5,1,2]

model32y = [5.4,4.9,4.6,4.6]
model32x = [0.5,1,2,2.5]

model128y = [5.8,4.8,5.3,5.4]
model128x = [0,0.5,1.0,2.0]



plt.plot(model8x, model8y, label="8")
plt.plot(model16x, model16y, label="16")
plt.plot(model32x, model32y, label="32")
plt.plot(model128x, model128y, label="128")

#plt.hlines(1.8, 1, len(x2), colors = "gray", linestyles = "dashed")
#plt.hlines(0.5, 1, len(x2), colors = "gray", linestyles = "dashed")

plt.ylabel("Distance value, k=10", fontsize=16)
plt.xlabel("Number of epochs", fontsize=16)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
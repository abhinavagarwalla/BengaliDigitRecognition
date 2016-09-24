import numpy as np
import matplotlib.pyplot as plt

f = open('../results/run1_2.txt').readlines()
f = [f[i].strip().split() for i in range(5,len(f),2)]
f = np.array(f)
X1 = f[:,6].astype(np.float32)
X2 = f[:,12].astype(np.float32)
plt.plot(range(len(X1)),X1,color='b')
plt.hold(True)
plt.plot(range(len(X2)),X2,color='r')
plt.show()

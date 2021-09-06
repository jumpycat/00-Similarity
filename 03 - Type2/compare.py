
import matplotlib.pyplot as plt




lsrm =r'logs-nt-c40-lsrm8.txt'
base = r'logs-nt-c40-lsrm6.txt'

lsrm_acc = []
base_acc = []
lsrm_loss =  []
base_loss =  []

with open(lsrm,'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        loss = data[3]
        acc = data[5][:-2]
        lsrm_loss.append(float(loss))
        lsrm_acc.append(float(acc))



with open(base,'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        loss = data[3]
        acc = data[5][:-2]
        base_loss.append(float(loss))
        base_acc.append(float(acc))

plt.ylim(0.3,1.0)
plt.plot(lsrm_acc[:500],color='r')
plt.plot(base_acc[:500],color='b')
plt.show()

import numpy as np
print(np.mean(lsrm_acc[:800]))
print(np.mean(base_acc[:800]))
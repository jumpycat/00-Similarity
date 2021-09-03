
import matplotlib.pyplot as plt




lsrm =r'logs-LSRM.txt'
base = r'logs-resnet18.txt'

lsrm_acc = []
base_acc = []
lsrm_loss =  []
base_loss =  []

with open(lsrm,'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        loss = data[3]
        acc = data[5][:-2]
        if float(acc)<=0.3:
            continue
        else:
            lsrm_loss.append(float(loss))
            lsrm_acc.append(float(acc))



with open(base,'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        loss = data[3]
        acc = data[5][:-2]
        if float(acc)<=0.3:
            continue
        else:
            base_loss.append(float(loss))
            base_acc.append(float(acc))

plt.ylim(0.3,1.1)
plt.plot(lsrm_acc,color='r')
plt.plot(base_acc,color='b')
plt.show()
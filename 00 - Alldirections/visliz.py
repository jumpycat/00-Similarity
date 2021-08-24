import matplotlib.pyplot as plt

data = []
with open("logs.txt", "r") as f:
    index = 0
    for line in f.readlines():
        if index%50 == 0:
            data.append(float(line[-6:]))
        index += 1

plt.plot(data)
plt.show()
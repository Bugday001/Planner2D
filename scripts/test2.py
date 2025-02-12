import numpy as np
import matplotlib.pyplot as plt

obs = []
state1 = np.random.randint(0, 10, size=2)
state2 = np.array([29, 28])
x1, y1 = state1[0], state1[1]
x2, y2 = state2[0], state2[1]
dy = y2 - y1
dx = x2 - x1
reverse = False
if abs(dy) > abs(dx):
    x1, y1 = y1, x1
    x2, y2 = y2, x2
    reverse = True
k = (y2 - y1) / (x2 - x1 + 1e-12)
if x1 > x2:
    step = -1
else:
    step = 1
for x in range(x1, x2 + 1, step):
    y = (x - x1) * k + y1
    if reverse:
        obs.append([int(y), x])
    else:
        obs.append([x, int(y)])

obs = np.array(obs)
plt.plot([x1, x2], [y1, y2])
plt.scatter(obs.T[0], obs.T[1])
plt.xticks(range(30))
plt.yticks(range(30))
plt.grid()
plt.show()

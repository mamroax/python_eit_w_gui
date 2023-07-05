from numpy import *
import math
import matplotlib.pyplot as plt

t = linspace(0,2*math.pi,400)
a = sin(t)
b = cos(t)
c = a + b
for i in range(10):
    plt.plot(t, a, t, b, t, c)
plt.show()

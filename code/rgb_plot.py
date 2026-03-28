import numpy as np
import matplotlib.pyplot as plt


r = [9.7, 10.4,10.2,10.1,11.1,12.3,13.7,18.7,27.6,55.6]
g = [20.7,23.2,26.7,31.8,40.4,56.7,83.6,146.1,289.3,792]
b = [7.8,8.1,8.2,8.8,9.8,12,16.3,25.3,44.5,109.1]
d = [20,18,16,14,12,10,8,6,4,2]



print("")


plt.figure()
plt.loglog(d ,r, label = "R", color = "red")
plt.loglog(d ,g, label = "G", color = "green")
plt.loglog(d ,b, label = "B", color = "blue")
plt.grid()
plt.xlabel("distance in cm")
plt.ylabel("rgb code")
plt.show()
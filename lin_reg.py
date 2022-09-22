
import matplotlib.pyplot as plt
import numpy as np    
import random
from time import sleep
def best_fit(x,y):
    slope, intercept = np.polyfit(x, y, 1)
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def rss(real_y, real_x, w1, w2):
    res = 0
    for x,y in zip(real_x, real_y):
        res += (y - (x*w1 + w2))**2
        
    return res
    

figure, ax = plt.subplots(figsize=(4,5))


x_real = [9.63, 5.26, 9.50, 9.35, 9.2, 7.67]
y_real = [7.525, 4.348, 7.382, 7.305, 6.538, 5.207]

# GUI

plt.ion()

#  Plot

plot1 = ax.scatter(x_real, y_real)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Labels

plt.xlabel("X-Axis",fontsize=18)
plt.ylabel("Y-Axis",fontsize=18)
w1 = 1
w2 = 1
best_state = (w1, w2)
residual_sum_of_squares = rss(x_real, y_real, w1, w2)
abline(w1, w2)
figure.canvas.draw()
figure.canvas.flush_events()
plt.show()
for i in range(100000):
    w1 = random.uniform(-5, 5)
    w2 = random.uniform(-5, 5)
    local_residual_sum_of_squares = rss(x_real, y_real, w1, w2)
    if local_residual_sum_of_squares < residual_sum_of_squares:
        best_state = (w1, w2)
        residual_sum_of_squares = local_residual_sum_of_squares
        abline(w2, w1)
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.show()
    
abline(best_state[1], best_state[0])
figure.canvas.draw()
figure.canvas.flush_events()
plt.show()
print("Done")
plt.show()
sleep(10)
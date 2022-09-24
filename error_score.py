from imports import *

    
def MAPE(y_actual, w1, w2):
    rss = 0
    for y_a, y_p in zip(y_actual, [w1*x + w2 for x in x_real]):
        rss += abs((y_a - y_p) / y_a)
    return (rss / (len(y_actual))) * 100.0


def MSE(y_actual, w1, w2):
    rss = 0
    for y_a, y_p in zip(y_actual, [w1*x + w2 for x in x_real]):
        rss += (y_a - y_p)**2
    return rss / len(y_actual)

error = MSE

x_real = [9.63,  5.26,  9.50,  9.35,  9.2,   7.67]
y_real = [7.525, 4.348, 7.382, 7.305, 6.538, 5.207]


w1 = 1
w2 = 1

best_w1, best_w2 = np.polyfit(x_real, y_real, 1)

lr = LinearRegression()
lr.fit(np.array(x_real).reshape(1,-1), np.array(y_real).reshape(1,-1))
lr_w1, lr_w2 = lr.coef_, lr.intercept_
print(lr_w1, '\n', lr_w2)
error_rand, error_best = error(y_real, w1, w2), error(y_real, best_w1, best_w2)
print("Error random:\t", error(y_real, w1, w2))
print("Error polyfit:\t", error(y_real, best_w1, best_w2))
print("Error LinReg:\t", error(y_real, lr_w1, lr_w2))



def plot_data():
    figure, ax = plt.subplots(figsize=(4,5))
    plt.ion()

    plot1 = ax.scatter(x_real, y_real)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.xlabel("X-Axis",fontsize=18)
    plt.ylabel("Y-Axis",fontsize=18)
    plt.show()
    
def plot_error():
    figure, ax = plt.subplots(figsize=(4,5))
    plt.ion()
    e = [error(y_real, w_1, w2) for w_1 in range(-10,10)]
    plot1 = ax.scatter([w_1 for w_1 in range(-10,10)], e)
    ax.set_xlim(-10, 10)

    plt.xlabel("X-Axis",fontsize=18)
    plt.ylabel("Y-Axis",fontsize=18)
    plt.show()
    plt.pause(54)
    
plot_data()
plot_error()
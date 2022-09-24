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

x_real = [1.63,  2.26,  3.50,  4.35,  5.2,   6.67]
y_real = [4.525, 5.348, 5.382, 3.305, 3.538, 5.207]


w1 = 1
w2 = 1

weights = np.polyfit(x_real, y_real, 3)
print(weights)

print("Error random:\t", error(y_real, w1, w2))



def plot_data():
    figure, ax = plt.subplots(figsize=(4,5))
    plt.ion()

    plot1 = ax.scatter(x_real, y_real)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.xlabel("X-Axis",fontsize=18)
    plt.ylabel("Y-Axis",fontsize=18)
    print("Starting")
    for x in x_real:
        print("Hello?", x)
        plt.plot(x, PolyCoefficients(x, weights))
    plt.show()
    plt.pause(5)
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

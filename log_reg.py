from math import e, log
from threading import local
from drawing import abline
from imports import *


def log_loss(data, w1, w2, w0):
    predicted_probability = [1.0/(1.0+(2.7**-(w1*price + w2 * quality + w0))) for price,quality,purchased in data]
    n = len(data)
    loss = 0.0
    for (price, quality, purchased), predicted in zip(data, predicted_probability): 
        if purchased == 'g':
            y = 1.0;
        else: y = 0.0
        if predicted == 1.0:
            loss += y * log(predicted)
        elif predicted == 0.0:
            loss += (1.0 - y)*log(1.0 - predicted);
        else:
            loss += y * log(predicted) + (1.0-y)*log(1.0-predicted)
            
    return -1.0/n * loss

data = []
count = 50
place_sep = 34;
for i in range(count):
    x, y = random.uniform(0, 50), random.uniform(0, 50)
    noise = random.uniform(0, 5)
    color = 'r'
    if x < place_sep - noise and y > place_sep - 2*noise:
        color = 'g'
    data.append((x,y,color))
price = [x for x,y,c in data]
quality = [y for x,y,c in data]
purchased = [c for x,y,c in data]
    

loss = 10000
best_w1 = 0
best_w2 = 0
best_w0 = 0
for i in range(100000):
    w0 = random.uniform(0, 50);
    w1 = random.uniform(1, 5);
    w2 = random.uniform(1, 5);
    local_loss = log_loss(data, w1, w2, w0)
    if (local_loss) < loss:
        loss = (local_loss)
        best_w1 = w1
        best_w2 = w2
        best_w0 = w0



print(best_w1, best_w2, best_w0, loss)


figure, ax = plt.subplots()
plt.ion()

plot1 = ax.scatter(price, quality, c=purchased)

x_vals = [x for x in range(50)]
p1s = [1.0/(1.0+(2.7**-(best_w1*price + best_w0))) for price,quality,purchased in data]
p2s = [1.0/(1.0+(2.7**-(best_w2 * quality + best_w0))) for price,quality,purchased in data]
p1s_odds = [1.0 if p == 1.0 else log(p /(1.0 - p)) for p in p1s]
p2s_odds = [1.0 if p == 1.0 else log(p /(1.0 - p)) for p in p2s]
print([p1-p2 for p1,p2 in zip(p1s_odds, p2s_odds)])
for p1,p2 in zip(p1s_odds, p2s_odds):
    abline(p1, p2)

ax.set_xlim(0, 50)
ax.set_ylim(0, 50)

plt.xlabel("X-Axis",fontsize=18)
plt.ylabel("Y-Axis",fontsize=18)
#plt.plot(price, [PolyCoefficients(x, weights) for x in price], color='r')
#plt.plot(price, [w1*x + w2 for x in price], color='g')
plt.show()
plt.pause(10)
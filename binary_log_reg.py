from math import e, log
from threading import local
from drawing import abline
from imports import *

def log_odds_to_probability(log_odd):
    return 2.7**log_odd / (1+2.7**log_odd)
    
def logit(p):
    if p == 0:
        return -10000
    if p == 1:
        return 10000
    return log(p / (1.0 - p))

def log_loss(data, w1, w0):
    predicted_probability = [1.0/(1.0+(2.7**-(w1*weight + w0))) for weight,obese in data]
    n = len(data)
    loss = 0.0
    for (weight, obese), predicted in zip(data, predicted_probability): 
        if obese == 'g':
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
    x = random.uniform(0, 50)
    noise = random.uniform(0, 10)
    color = 'r'
    if x < place_sep - noise:
        color = 'g'
    data.append((x,color))
weight = [x for x,c in data]
obese = [c for x,c in data]
obese_prob = [0 if c == 'g' else 1 for c in obese]
    
    
transformed_probabilities = [logit(y) for y in obese_prob]

loss = 10000
best_w1 = 0
best_w0 = 0
for i in range(100000):
    w0 = random.uniform(0, 50);
    w1 = random.uniform(1, 5);
    local_loss = log_loss(data, w1, w0)
    if (local_loss) < loss:
        loss = (local_loss)
        best_w1 = w1
        best_w0 = w0



print(best_w1, best_w0, loss)

def plot_original_data():
    figure, ax = plt.subplots()
    plt.ion()

    plot1 = ax.scatter(weight, obese_prob, c=obese)

    plt.xlabel("Weight",fontsize=18)
    plt.ylabel("Obese?",fontsize=18)
    #plt.plot(weight, [PolyCoefficients(x, weights) for x in weight], color='r')
    #plt.plot(weight, [w1*x + w2 for x in weight], color='g')
    plt.show()

    # print([x for x in range(50)], [logit(y) for y in p1s])
    # ax.scatter([x for x in range(50)], [logit(y) for y in p1s], c = 'b')

def plot_transformed_data():
    figure, ax = plt.subplots()
    plt.ion()

    plot1 = ax.scatter(weight, transformed_probabilities, c=obese)

    plt.xlabel("Weight",fontsize=18)
    plt.ylabel("Obese?",fontsize=18)
    #plt.plot(weight, [PolyCoefficients(x, weights) for x in weight], color='r')
    #plt.plot(weight, [w1*x + w2 for x in weight], color='g')
    plt.show()

    # print([x for x in range(50)], [logit(y) for y in p1s])
    # ax.scatter([x for x in range(50)], [logit(y) for y in p1s], c = 'b')
plot_transformed_data()
plt.pause(10)
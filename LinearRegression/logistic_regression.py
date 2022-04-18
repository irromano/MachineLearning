
import numpy as np
import matplotlib.pyplot as plt


# generating a dataset
np.random.seed(2)
n = 3000
n_train = 2000
n_test = 1000
x = np.zeros((2, n))
y = np.zeros(n)
p1 = 0.4
p2 = 0.6
for i in range(n):
    y[i] = np.random.choice([0, 1], 1, p=[p1, p2])
    if y[i] == 1:
        # if i%2==0:
        mean1 = [1, 0.5]
        cov1 = [[3, -1], [-1, 1.5]]
        x[:, i] = np.random.multivariate_normal(mean1, cov1, 1)
    else:
        mean2 = [-1, -3]
        cov2 = [[2, 0], [0, 1]]
        x[:, i] = np.random.multivariate_normal(mean2, cov2, 1)

plt.plot(x[0, y == 0], x[1, y == 0], 'rx')
plt.plot(x[0, y == 1], x[1, y == 1], 'o')
plt.axis('equal')
plt.show()


# train test split
x = np.vstack([np.ones((1, n)), x])
x_train = x[:, :n_train]
x_test = x[:, n_train:]
y_train = y[:n_train]
y_test = y[n_train:]


# calculating the gradient
def sigmoid(theta, xi):
    return 1/(1+np.exp(-np.inner(theta, xi)))


def grad(theta, x_train, y_train):
    r = 0  # regularization parameter
    gradient = 2*r*theta
    hessian = 2*r*np.eye(x.shape[0])
    n = x_train.shape[1]
    for i in range(n):
        current_data = x_train[:, i]
        gradient += (current_data*(sigmoid(theta, current_data)-y_train[i]))
        hessian += np.outer(current_data, current_data) * \
            (sigmoid(theta, current_data))*(1-sigmoid(theta, current_data))
    return gradient, hessian


# running the gradeient decent
theta = np.zeros((x.shape[0],))
alpha = 0.01
for i in range(50):
    g, h = grad(theta, x_train, y_train)
    #theta -= np.linalg.inv(h).dot(g)
    theta -= alpha*g


# Training Accuracy
prob = []
for i in range(n_train):
    current_data = x_train[:, i]
    prob.append(sigmoid(theta, current_data))
prob = np.array(prob)

y_pred = (prob > 0.5)*1.0

print(sum(y_pred == y_train)/n_train)


# prediction and test accuracy

prob = []
for i in range(n_test):
    current_data = x_test[:, i]
    prob.append(sigmoid(theta, current_data))
prob = np.array(prob)

y_pred = (prob > 0.5)*1.0

print(sum(y_pred == y_test)/n_test)


# decision boundry
x_0min = -5
x_0max = 5

x_1min = -5
x_1max = 5

Label0 = [[], []]

Label1 = [[], []]
x0 = np.linspace(x_0min, x_0max, 500)
x1 = np.linspace(x_1min, x_1max, 500)

for i in x0:
    for j in x1:
        if sigmoid(theta, np.array([1, i, j])) >= 0.5:
            Label0[0].append(i)
            Label0[1].append(j)
        else:
            Label1[0].append(i)
            Label1[1].append(j)

plt.plot(Label0[0], Label0[1], 'rx')
plt.plot(Label1[0], Label1[1], 'o')
plt.tight_layout()
plt.show()

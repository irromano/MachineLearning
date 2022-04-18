
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


#############################################################################
# Preparing the MNIST dataset for logistic regression
train_digits = sio.loadmat('data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']
data_y = np.zeros([num_train_images,10])
for lambda1 in range(num_train_images):
	data_y[lambda1,labels[lambda1][0]-1] = 1

print(data_x.shape)
print(data_y.shape)

#############################################################################
# The logistic function
def logistic(x,beta):
    y_pred = np.matmul(x,beta)
    logistic_prob = 1/(1 + np.exp(-y_pred))
    return logistic_prob

#############################################################################
# Here you create you own LOGISTIC REGRESSION algorith for the logistic regression model
# Hint: please check the slides of the logistic regression 
#############################################################################
def logistic_regression(beta, lr, x_batch, y_batch,lambda1):
	beta_next = beta + lr * (np.matmul(np.transpose(x_batch), (y_batch - logistic(x_batch, beta))) + lambda1 * beta)
	cost_batch = np.matmul(x_batch, beta_next)
	cost_batch[cost_batch > 0] = 1.0
	cost_batch[cost_batch < 0] = 0.0
	cost = np.sum(np.square(cost_batch - y_batch))
    
	return cost, beta_next

#############################################################################
 
def classifcation_ratio(prob_batch,y_batch):
	ratio = 0
	class_batch = np.argmax(prob_batch, axis=1)
	real_batch = np.argmax(y_batch, axis=1)
	true_result = class_batch == real_batch
	ratio = float(np.sum(true_result))/float(np.sum(true_result.size))
	return ratio

#############################################################################
#                       Main Function
#############################################################################

iterations = 21
Lambdas = np.linspace(0, 20, iterations)
accuracy = np.zeros(Lambdas.size)

for lambda1 in range(iterations):
	ratios = np.zeros(10)

	# Hyper-parameters
	training_epochs = 100
	learning_rate = 0.0005          # The optimization initial learning rate
	cost = 0

	probabilities = [np.zeros(10) for _ in range(data_y.shape[0])]
	probabilities = np.array(probabilities)
	for j in range(10):
		current_label = data_y[:,j]
		beta = np.random.randn(dim_images + 1)
		for epoch in range(training_epochs):
			cost, beta_next = logistic_regression(beta,learning_rate,data_x,current_label, lambda1)
			beta = beta_next
		probabilities[:, j] = logistic(data_x,beta)
	ratio = classifcation_ratio(probabilities, data_y)
	print(f'Lambda {lambda1}, the classification accuracy is {ratio*100:.2f}')
	accuracy[lambda1] = ratio



#############################################################################
# Visualizing five images of the MNIST dataset
# fig, _axs = plt.subplots(nrows=1, ncols=5)
# axs = _axs.flatten()

# for i in range(5):
# 	axs[i].grid(False)
# 	axs[i].set_xticks([])
# 	axs[i].set_yticks([])
# 	image = data_x[i*10,:784]
# 	image = np.reshape(image,(28,28))
# 	aa = axs[i].imshow(image,cmap=plt.get_cmap("gray"))

# fig.tight_layout()
# plt.show()

#############################################################################
# Visualizing classification accuracy across lambda
plt.xlabel("Lambda Value")
plt.ylabel("Classification Accuracy")
plt.plot(Lambdas, accuracy)
plt.show()


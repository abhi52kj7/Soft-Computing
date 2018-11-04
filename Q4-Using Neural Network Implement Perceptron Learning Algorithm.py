# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    if activation > 0:
        return 1
    elif activation == 0:
        return 0
    else:
        return-1

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            prediction = predict(row, weights)
            if prediction != row[-1]:
                error = row[-1]
            else:
                error = 0
            sum_error += (row[-1] - prediction)**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('Epoch = %d Error = %.3f'%(epoch + 1,sum_error))
    return weights

# Calculate weights
dataset = [[2.7810836,2.550537003,-1],
[1.465489372,2.362125076,-1],
[3.396561688,4.400293529,-1],
[1.38807019,1.850220317,-1],
[3.06407232,3.005305973,-1],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 4
print( 'Learning rate = %.3f, Epochs = %d'% (l_rate, n_epoch))
weights = train_weights(dataset, l_rate, n_epoch)
print('Weights =', weights)

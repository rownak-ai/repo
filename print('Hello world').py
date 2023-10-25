'''softmax_outputs = [[0.7,0.1,0.2],
                   [0.1,0.5,0.4],
                   [0.02,0.9,0.08]]

class_targets = [0,1,1]

for tar_index, distribution in zip(class_targets,softmax_outputs):
    print(distribution,tar_index)
    print(distribution[tar_index])'''
##Comments will be added soon

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,weight_regularization_L1=0, weight_regularization_L2=0, bias_regularization_L1=0, bias_regularization_L2=0):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularization_L1 = weight_regularization_L1
        self.weight_regularization_L2 = weight_regularization_L2
        self.bias_regularization_L1 = bias_regularization_L1
        self.bias_regularization_L2 = bias_regularization_L2

    def forward(self, inputs):

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularization_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += dL1*self.weight_regularization_L1

        if self.weight_regularization_L2 > 0:
            self.dweights += 2*self.weight_regularization_L2*self.weights

        if self.bias_regularization_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularization_L1*dL1

        if self.bias_regularization_L2 > 0:
            self.dbiases += 2*self.bias_regularization_L2*self.biases


        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:

    def __init__(self,rate):
        self.rate = 1-rate
        
    def forward(self,input):
        self.input = input
        self.binary_mask = np.random.binomial(1,self.rate,size=input.shape)/self.rate
        self.output = self.binary_mask*input

    def backward(self,dvalues):
        dinputs = dvalues*self.binary_mask
    
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0 ] = 0


class Activation_sigmoid:

    def forward(self,inputs):
        self.inputs = inputs
        self.output = 1/1+np.exp(inputs)

    def backward(self,dvalues):
        self.dvalues = self.output*(1-self.output)*dvalues


class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):

            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Optimizer_SGD:

    def __init__(self,learning_rate=1,decay=0,momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum*layer.weight_momentums - self.current_learning_rate*layer.dweights

            layer.weight_momentums = weight_updates

            biases_update = self.momentum*layer.bias_momentums - self.current_learning_rate*layer.dbiases

            layer.bias_momentums = biases_update

        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            biases_update = -self.current_learning_rate*layer.dbiases

        layer.weights += weight_updates
        layer.biases += biases_update

    def post_update_params(self):
        self.iterations += 1


class RMSprop:
    def __init__(self,learning_rate=0.001,decay=0,epsilon=1e-7,rho=0.99):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iteration = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iteration))

    def params_update(self,layer):
        if not hasattr(layer,'weight_cach'):
            layer.weight_cach = np.zero_like(layer.weights)
            layer.biases_cach = np.zero_like(layer.biases)

        layer.weight_cach = self.rho*layer.weight_cach + (1-self.rho)*(layer.dweights)**2
        layer.biases_cach = self.rho*layer.biases_cach + (1-self.rho)*(layer.dbiases)**2

        layer.weights += -self.current_learning_rate * layer.dweights/(np.sqrt(layer.weight_cach)+self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases/(np.sqrt(layer.biases_cach)+self.epsilon)

    def post_params_update(self):
        self.iteration += 1


class Adam:

    def __init__(self,learning_rate=0.001,decay=0.,beta_1=0.9,beta_2=0.999,epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.iteration))

    def update_params(self,layer):
        if not hasattr(layer,'weight_cach'):
            layer.weight_moments_cach = np.zeros_like(layer.weights)
            layer.weight_cach = np.zeros_like(layer.weights)
            layer.biases_moments_cach = np.zeros_like(layer.biases)
            layer.biases_cach = np.zeros_like(layer.biases)
        
        layer.weight_moments_cach = self.beta_1*layer.weight_moments_cach + (1-self.beta_1)*layer.dweights
        layer.biases_moments_cach = self.beta_1*layer.biases_moments_cach + (1-self.beta_1)*layer.dbiases

        weight_moments_corrected = layer.weight_moments_cach / (1-self.beta_1 ** (self.iteration+1))
        biases_moments_corrected = layer.biases_moments_cach / (1-self.beta_1 ** (self.iteration+1))

        layer.weight_cach = self.beta_2*layer.weight_cach + (1-self.beta_2)*layer.dweights**2
        layer.biases_cach = self.beta_2*layer.biases_cach + (1-self.beta_2)*layer.dbiases**2

        weight_cach_corrected = layer.weight_cach / (1-self.beta_2 ** (self.iteration+1))
        biases_cach_corrected = layer.biases_cach / (1 - self.beta_2 ** (self.iteration+1))

        layer.weights += -self.current_learning_rate*weight_moments_corrected/(np.sqrt(weight_cach_corrected)+self.epsilon)
        layer.biases += -self.current_learning_rate*biases_moments_corrected/(np.sqrt(biases_cach_corrected)+self.epsilon)

    def post_update_params(self):
        self.iteration +=1

class Loss:
    def regularization_loss(self,layer):
        regularization_loss = 0

        if layer.weight_regularization_L1 > 0:
            regularization_loss += layer.weight_regularization_L1 * np.sum(np.abs(layer.weights))
        
        if layer.weight_regularization_L2 > 0:
            regularization_loss += layer.weight_regularization_L2 * np.sum(layer.weights*layer.weights)

        if layer.bias_regularization_L1 > 0:
            regularization_loss += layer.bias_regularization_L1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularization_L2 > 0:
            regularization_loss += layer.bias_regularization_L2 * np.sum(layer.biases*layer.biases)

        return regularization_loss
    
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1
        self.dinputs = self.dinputs / samples

class Loss_Binary_Cross_Entropy(Loss):
    def forward(self,y_true,y_pred):
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        sample_loss = -(y_true*np.log(y_pred_clipped) + (1-y_true)*np.log(1-y_pred_clipped))
        sample_loss = np.mean(sample_loss,axis=-1)
        return sample_loss
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_values = np.clip(dvalues,1e-7,1-1e-7)

        self.inputs = -(y_true/clipped_values - (1-y_true)/(1-clipped_values))/outputs
        self.inputs = self.inputs/samples

class Mean_absolute_error(Loss):
    def forward(self,y_true,y_pred):
        pass

    def backward(self,dvalues,y_true):
        pass
    


X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,512,weight_regularization_L2=5e-4,bias_regularization_L2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#optimizer = Optimizer_SGD(decay=1e-3,momentum=0.5)
optimizer = Adam(learning_rate=0.02,decay=5e-7)

#precision = np.std(y)/250

for epoch in range(10001):


    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output,y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    #print(loss_activation.output[:5])
    #print('loss:', loss)
    prediction = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(prediction==y)

    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
        

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.inputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100,classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y_test)

prediction = np.argmax(loss_activation.output,axis=1)

if y_test.shape == 2:
    y_test = np.argmax(y_test,axis=1)

accuracy = np.mean(y_test==prediction)

print(f'validation accuracy: {accuracy:.3f}, loss: {loss:.3f}')
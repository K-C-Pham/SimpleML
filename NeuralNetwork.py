#import required, built in modules
import math
import random

#create the Neural Network as a class - 6 methods, 4 private, 2 public
class NeuralNetwork():
    def __init__(self):

        #Seed for RNG, so that we start with the same numbers each time
        random.seed(1)

        #Create 3 weights and set them to 3 random values between -1 and 1
        self.weights = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
    
    #Method makes a prediction based on the sum of weighted inputs, mapped to the sigmoid function
    def think(self, neuron_inputs):
        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

    #Adjust weights to minimize error
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            for training_set_example in training_set_examples:

                #predict output based on training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                #calculate the error as difference between desired output and predicted output
                error_in_output = training_set_example["output"] - predicted_output

                #Iterate through weights and adjust each one. Self.weights returns the 3 weights. Index will loop through those - 0, 1, 2.
                for index in range(len(self.weights)):

                    #get neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    #calculate weight adjustment using delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    #adjust weight
                    self.weights[index] += adjust_weight


    #calculate the sigmoid activation function
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1/(1+math.exp(-sum_of_weighted_inputs))
    
    #calculate gradient of sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):
        return neuron_output * (1-neuron_output)

    #multiply each input by its weight, then sum all 3 weighted inputs
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for i, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[i] * neuron_input
        return sum_of_weighted_inputs

#initialize the class
neural_network = NeuralNetwork()

print("Random starting weights " + str(neural_network.weights))

#create training data as a list variable. Each training set is a dictionary, input:[list of data], output: number
training_set_example = [{"inputs":[0,0,1],"output":0},
                        {"inputs":[1,1,1],"output":1}, 
                        {"inputs":[1,0,1],"output":1}, 
                        {"inputs":[0,1,1],"output":0}]

# Train the neural network using 10,000 iterations
neural_network.train(training_set_example,number_of_iterations=10000)

print("New weights after training: " + str(neural_network.weights))

# Make a prediction for a new situation
new_situation = [1,0,0]
prediction = neural_network.think(new_situation)

print(f"Prediction for the new situation {new_situation} -> ?")
print(str(prediction))
#!/usr/bin/env python3

import numpy as np
import csv

class LinearModelBase:
    
    def __init__(self, lr=0.01, epochs=50, rand_seed=1):
        
        self.lr = lr #Initializes learning rate of instance (default of 0.01)
        self.epochs = epochs #Initializes epoch count of instance (default of 50)
        self.rand_seed = rand_seed #Initializes random seed value to ensure consistent random gen output


    def initialize_weights(self, t_data):

        rand_gen = np.random.default_rng(self.rand_seed)  #Creates a random number generator
        self.weights_ = rand_gen.normal(loc=0.0, scale=0.01, size=1 + t_data.shape[1]) 
        ''' 
        Generates random weight values for this instance based on a normal dist. with a mean 
        of 0 & sd of 0.01 for a numpy vector of equal size to the number of columns in the 
        dataset plus an additional column to store the initial bias term
        '''
    
    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0] #Dot product of weight & dataset plus bias term
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1) #Returns an array of predicted values based on whether net input is less than or greater than zero
    
    def check_accuracy(self, results, targets):
        correct = np.sum(results == targets)
        return correct / len(targets)


class Adaline(LinearModelBase):
    def fit(self, t_data, targets):

        self.initialize_weights(t_data)

        self.costs_ = [] #Initializes instance field to store cost value of each epoch iteration

        for i in range(self.epochs):
            net_input = self.net_input(t_data) #Returns an array of net input result of each row
            output = self.activation(net_input) #For Adaline, returns the net input vector unchanged (identity activation)
            errors = (targets - output) #Creates a vector to store the difference/error of each row
            self.weights_[1:] += self.lr * t_data.T.dot(errors) #Updates the weight by multiplying lr by error by feature value & adding to original weight value
            self.weights_[0] += self.lr * errors.sum() #Updates the bias by multiplying lr by the sum of all sample errors & adding to current bias value
            cost = (errors ** 2).sum() / 2.0 #Calculates the cost/loss of this epoch iteration
            self.costs_.append(cost) #Adds this epoch's cost value to the costs vector
        return self
    
class Perceptron(LinearModelBase):
    def fit(self, t_data, targets):

        self.initialize_weights(t_data)

        for i in range(self.epochs):
            for xi, target in zip(t_data, targets):
                update = self.lr * (target - self.predict(xi)) #Calculates error times lr
                self.weights_[1:] += update * xi #Accounts for multiplying by xi before incrementing the weights
                self.weights_[0] += update #Since x0 would equal 1, updates bias without needing x0
        return self
            

    
def csv_to_numpy(file):
    return np.genfromtxt(file, delimiter=',', skip_header=1)
    
if __name__ == "__main__":
    td_file = input("What is the name of the csv file you'd like to use for training (Must be located in program's directory & have targets as last entry)? \n")
    t_data = csv_to_numpy(td_file)

    X = t_data[:, :-1]
    y = t_data[:, -1]

    model_choice = input("Would you like to use an adaline or perceptron model?(adaline/perceptron): ").lower()

    if model_choice not in ['adaline', 'perceptron']:
        raise ValueError("Unknown Model Type Selected")
    
    lr = float(input("What learning rate would you like to use? (0.0-1.0): "))
    if not (0.0 <= lr <= 1.0):
        raise ValueError("Learning Rate must be between 0.0 & 1.0")
    
    epochs = int(input("How many epochs?: "))
    if epochs < 1:
        raise ValueError("Number of epochs must be greater than or equal to 1.")
    
    if model_choice == 'adaline':
        model = Adaline(lr, epochs)
    else:
        model = Perceptron(lr, epochs)
    
    model.fit(X, y)

    p_file = input("What is the name of the csv file you'd like to use for testing (Must be located in program's directory)? \n")
    p_data = csv_to_numpy(p_file)
    target_location = int(input("Please enter the column number of the target values or 0 if they are not included in the dataset: "))

    if target_location == 0:
        PX = p_data
        predictions = model.predict(PX)
    else:
        target_location += -1
        PX = np.delete(p_data, target_location, axis=1)
        PY = p_data[:, target_location]
        predictions = model.predict(PX)
        accuracy = model.check_accuracy(predictions, PY)
        print(f"Accuracy Rate: {accuracy * 100:.2f}%")
    
    record_output = input("Would you like to save the model output in a seperate file?(y/n): ").lower().strip()

    if record_output == 'n':
        print("Goodbye!")
    elif record_output != 'y':
        raise ValueError("Invalid output, must be 'y' or 'n'")
    else:
        ofile_name = input("Please input the proposed filename (without extension): ") + ".csv"
        id_col = int(input("Which row of the csv file used for prediction is the sample's unique identifer located? (Starting at 1): ")) - 1
        with open(p_file, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            id_name = headers[id_col]

            unique_ids = []
            for row in reader:
                unique_ids.append(row[id_col])


        if len(unique_ids) != len(predictions):
            raise ValueError("Number of predictions does not match the number of samples provided.")

        with open(ofile_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id_name, "Prediction"])
            
            for uid, prediction in zip(unique_ids, predictions):
                writer.writerow([uid, prediction])
    
        print(f"Predictions saved to {ofile_name}")

    
    

#!/usr/bin/env python3

import numpy as np
import csv

class LinearModelBase:
    
    def __init__(self, lr=0.01, epochs=50, rand_seed=1):
        
        self.lr = lr # Initializes learning rate of instance (default of 0.01)
        self.epochs = epochs # Initializes epoch count of instance (default of 50)
        self.rand_seed = rand_seed # Initializes random seed value to ensure consistent random gen output

    def set_is_adaline(self, is_adaline):
        
        self.is_adaline_ = is_adaline # Sets uninitialized model type of instance


    def initialize_weights(self, t_data):
        
        rand_gen = np.random.default_rng(self.rand_seed) # Initializes a new random number generator using the instance's seed value
        self.weights_ = rand_gen.normal(loc=0.0, scale=0.01, size=1 + t_data.shape[1]) # Initializes the instance's weight values & bias value with a random number from a normal dist. of mean 0 & std. dev 0.01
    
    def net_input(self, X):
        return np.dot(X, self.weights_[1:].T) + self.weights_[0] # Returns dot product of the transposed weight vector and the vector/matrix (based on instance model) summed with the bias value

    
    def activation(self, X):
        if self.is_adaline_:    
            return X # Adaline uses an identity activation function
        else:
            return np.where(X >= 0.0, 1, 0) # Perceptron dosen't really have an activation function; uses a step function
    
    def predict(self, X):
        if self.is_adaline_:
            return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0) # Returns an array of predicted values based on whether net input is less than or greater than zero
        else:
            return self.activation(self.net_input(X)) # Returns the activation step value output of the net input value of a single row
    
    def check_accuracy(self, results, targets):
        correct = np.sum(results == targets) # Keeps tally of how many results equal the target's values
        return correct / len(targets) # Returns float value of dividing correct predictions by number of targets
    
    def log_weights(self, weights):
        
        with open('weights.csv', mode='a', newline='') as file: # Writes the weight values to weights.csv where every line represents an epoch
            writer = csv.writer(file)
            writer.writerow(weights)

    def clear_weights_log(self):
        with open('weights.csv', mode='w') as file: # Clears weights.csv to reset before training a new model
            pass



class Adaline(LinearModelBase):
    def fit(self, t_data, targets):

        self.initialize_weights(t_data) # Initializes a corresponding number of weights to the number of features in a dataset
        self.clear_weights_log() # Clears weights.csv data before training a new instance


        self.costs_ = [] #Initializes instance field to store cost value of each epoch iteration

        for i in range(self.epochs):
            net_input = self.net_input(t_data) #Returns an array of net input result of each row
            output = self.activation(net_input) #For Adaline, returns the net input vector unchanged (identity activation)
            errors = (targets - output) #Creates a vector to store the difference/error of each row
            self.weights_[1:] += self.lr * t_data.T.dot(errors) #Updates the weight by multiplying lr by error by feature value & adding to original weight value
            self.weights_[0] += self.lr * errors.sum() #Updates the bias by multiplying lr by the sum of all sample errors & adding to current bias value
            self.log_weights(self.weights_)
            cost = (errors ** 2).sum() / 2.0 #Calculates the cost/loss of this epoch iteration
            self.costs_.append(cost) #Adds this epoch's cost value to the costs vector
        return self
    
class Perceptron(LinearModelBase):
    def fit(self, t_data, targets):

        self.initialize_weights(t_data)
        self.clear_weights_log()

        for i in range(self.epochs):
            for xi, target in zip(t_data, targets):
                update = self.lr * (target - self.predict(xi)) #Calculates error times lr
                self.weights_[1:] += update * xi #Accounts for multiplying by xi before incrementing the weights
                self.weights_[0] += update #Since x0 would equal 1, updates bias without needing x0
            self.log_weights(self.weights_)
        return self
            

    
def csv_to_numpy(file, titanic):
    if titanic:
        dtype = [
            ('PassengerId', 'f8'),
            ('Survived', 'f8'),
            ('Pclass', 'f8'), 
            ('LastName', 'U50'),
            ('FirstName', 'U50'),
            ('Sex', 'U6'),
            ('Age', 'f8'),
            ('SibSp', 'f8'),
            ('Parch', 'f8'),
            ('Ticket', 'U20'),
            ('Fare', 'f8'),
            ('Cabin', 'U20'),
            ('Embarked', 'U1')
        ]
        
        return np.genfromtxt(file, delimiter=',', skip_header=1, dtype=dtype, missing_values='', filling_values=np.nan)
    else:
        return np.genfromtxt(file, delimiter=',', skip_header=1)
    


def get_column_numbers(prompt):
    user_input = input(prompt)
    if user_input.strip() == "":
        return []
    else:
        return [int(col) - 1 for col in user_input.split()]
    
def process_data(t_data, titanic):
    if titanic:
        target_col = input("\n\nEnter the column header name of the target values w/ proper punctuation (Press Enter for default): ")
        if target_col == '':
            target_col = 'Survived'
        

        unique_id_col = input("\n\nEnter the column header name of the unique IDs w/ proper punctuation (Press Enter for default): ")
        if unique_id_col == '':
            unique_id_col = 'PassengerId'


        irrelevant_cols = get_column_numbers("\n\nEnter column header names of fields you want to exclude from training seperated by spaces w/ proper punctuation (Press Enter for default): ")
        if not irrelevant_cols:
            irrelevant_cols = ['LastName', 'FirstName', 'Ticket', 'Cabin']

        features = [col for col in t_data.dtype.names if col not in irrelevant_cols + [target_col, unique_id_col]]

        X = np.column_stack([t_data[field] for field in features])
        if X.ndim == 1:
            X = X.reshape(-1, 7)
        Y = np.asarray(t_data[target_col])
        Y = Y + 1
        X = conversions(X)
        X = X.astype(np.float64)
        X = standardize(X)


        
    else:
        target_col = int(input("\n\nEnter the column number (starting at 1) of the target values: ")) - 1

        unique_id_col = int(input("\n\nEnter the column number (starting at 1) of the unique IDs: ")) - 1

        irrelevant_cols = get_column_numbers("\n\nEnter column numbers (starting at 1) to exclude from the training dataset seperated by spaces, or press Enter for none: ")

        irrelevant_cols = [col - 1 for col in irrelevant_cols]

        remove_columns = set([target_col, unique_id_col] + irrelevant_cols)

        X = np.delete(t_data, list(remove_columns), axis=1)
        Y = t_data[:, target_col]
        Y = Y + 1

    return X, Y, target_col, unique_id_col, irrelevant_cols

def conversions(X):
    # Convert 'Sex' from 'male', 'female' to 1, 2
    X[:, 1] = np.where(X[:, 1] == 'male', 1, np.where(X[:, 1] == 'female', 2, np.nan))

    # Convert 'Pclass'
    X[:, 0] = np.where(X[:, 0] == 3, 1, np.where(X[:, 0] == 2, 2, 3))
    
    # Handle 'Embarked': Replace missing or invalid values with NaN and then convert to numeric
    valid_embarked = ['C', 'Q', 'S']
    X[:, 6] = np.where(np.isin(X[:, 6], valid_embarked), X[:, 6], 'Q')

    c = 3.0
    q = 2.0
    s = 1.0
    default = 2.0


    # Convert 'Embarked' to numeric values
    X[:, 6] = np.where(X[:, 6] == 'C', c, np.where(X[:, 6] == 'Q', q, np.where(X[:, 6] == 'S', s, default)))

    X = X.astype(np.float64)

    # Increment 'SibSp' and 'Parch'
    X[:, 3] = X[:, 3] + 1
    X[:, 4] = X[:, 4] + 1

    # Convert 'Embarked' to float and calculate the average 'Embarked' value (ignoring NaN values)
    X[:, 6] = X[:, 6].astype(float)

    # Handle missing 'Age' values
    mean_age = np.nanmean(X[:, 2].astype(float))
    X[:, 2] = np.where(np.isnan(X[:, 2]), mean_age, X[:, 2])

    return X


def standardize(X):
    for i in range(7):
        col_min = np.nanmin(X[:, i])
        col_max = np.nanmax(X[:, i])

    if col_min != col_max:
        X[:, i] = (X[:, i] - col_min) / (col_max - col_min)

    else:
        X[:, i] = 0
    
    return X

def run_program():
    print("\n\n************** COMP 379 HW2 | Lueders, Sebastian **************\n\n\n")
    
    
    td_file = input("What is the name of the csv file you'd like to use for training (Must be located in program's directory)?: ")


    if td_file == 'titanic_train.csv' or td_file == 'titanic_test.csv':
        titanic = True
        t_data = csv_to_numpy(td_file, titanic)
    else:
        titanic = False
        t_data = csv_to_numpy(td_file, titanic)


    TX, TY, target_col, unique_id_col, irrelevant_cols = process_data(t_data, titanic)

    model_choice = input("\n\nWould you like to use an adaline or perceptron model?(adaline/perceptron): ").lower()

    if model_choice not in ['adaline', 'perceptron']:
        raise ValueError("Unknown Model Type Selected")
    
    lr = float(input("\n\nWhat learning rate would you like to use? (0.0-1.0): "))
    if not (0.0 <= lr <= 1.0):
        raise ValueError("Learning Rate must be between 0.0 & 1.0")
    
    epochs = int(input("\n\nHow many epochs?: "))
    if epochs < 1:
        raise ValueError("Number of epochs must be greater than or equal to 1.")
    
    if model_choice == 'adaline':
        model = Adaline(lr, epochs)
        model.set_is_adaline(True)
    else:
        model = Perceptron(lr, epochs)
        model.set_is_adaline(False)
    
    print("\nTraining Model...")
    model.fit(TX, TY)

    p_file = input("\n\n\nWhat is the name of the csv file you'd like to use for testing (Must be located in program's directory & non-target columns must be ordered exactly the same as the training dataset)? \n")
    p_data = csv_to_numpy(p_file, titanic)
    
    

    if titanic:
        features = [col for col in p_data.dtype.names if col not in irrelevant_cols + [target_col, unique_id_col]]

        PX = np.column_stack([p_data[field] for field in features])
        if PX.ndim == 1:
            PX = PX.reshape(-1, 7)
        PY = np.asarray(p_data[target_col])
        PX = conversions(PX)
        PX = PX.astype(np.float64)
        PX = standardize(PX)
        predictions = model.predict(PX)
        accuracy = model.check_accuracy(predictions, PY)
        print(f"\n\n*** Accuracy Rate: {accuracy * 100:.2f}%")
    else:
        targets_present = input("\n\nAre the target values saved within a column in this dataset(yes/no): ").lower().strip()

        if targets_present == "no":
            targets_present = 0
        elif targets_present == "yes":
            targets_present = 1
        else:
            raise ValueError("Input for target value existance must be 'yes' or 'no'")

        if targets_present == 0:
            if (unique_id_col > target_col):
                unique_id_col += -1
            
            for col in irrelevant_cols:
                if col > target_col:
                    col += -1
            
            columns_to_remove = set([unique_id_col] + irrelevant_cols)
            
            PX = np.delete(p_data, list(columns_to_remove), axis=1)
            predictions = model.predict(PX)
        else:
            columns_to_remove = set([target_col, unique_id_col] + irrelevant_cols)
            PX = np.delete(p_data, list(columns_to_remove), axis=1)
            PY = p_data[:, target_col]
            predictions = model.predict(PX)
            accuracy = model.check_accuracy(predictions, PY)
            print(f"\n\n*** Accuracy Rate: {accuracy * 100:.2f}%")
    
    
    
    record_output = input("\n\nWould you like to save the model output in a seperate file?(y/n): ").lower().strip()

    if record_output == 'n':
        print("\nExiting Program...\n\n")
    elif record_output != 'y':
        raise ValueError("Invalid output, must be 'y' or 'n'")
    else:
        ofile_name = input("\n\nPlease input the proposed filename (without extension): ") + ".csv"
        id_col = int(input("\n\nWhich header represents the sample's unique identifer in the testing dataset? (Starting at 1): ")) - 1
        target_col = int(input("\n\nWhich header represents the sample's target value in the testing dataset? (Starting at 1): ")) - 1
        with open(p_file, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            id_name = headers[id_col]
            target_name = headers[target_col]

            unique_ids = []
            target_values = []
            for row in reader:
                unique_ids.append(row[id_col])
                target_values.append(row[target_col])


        if len(unique_ids) != len(predictions) or len(unique_ids) != len(target_values):
            raise ValueError("Number of predictions does not match the number of samples provided.")

        with open(ofile_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id_name, "Prediction", "Target"])
            
            for uid, prediction, target in zip(unique_ids, predictions, target_values):
                writer.writerow([uid, prediction, target])
    
        print(f"\n\nResults saved to {ofile_name}")
    
if __name__ == "__main__":
    run_program()

    
    

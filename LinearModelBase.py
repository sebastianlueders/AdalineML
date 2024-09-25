#!/usr/bin/env python3

import numpy as np
import csv
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import os


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

    
    def activation(self, X, baseline=False):
        
        if baseline ==True:
            return np.where(X >= 0.0, 1, 0)
        elif self.is_adaline_:    
            return X # Adaline uses an identity activation function
        else:
            return np.where(X >= 0.0, 1, 0) # Perceptron dosen't really have an activation function; uses a step function
    
    def predict(self, X, baseline=False):
        
        if baseline == True:
            return self.activation(self.net_input(X), baseline)
        elif self.is_adaline_:
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

        self.initialize_weights(t_data) # Initializes a corresponding number of weights to the number of features in a dataset
        self.clear_weights_log() # Clears weights.csv data before training a new instance

        for i in range(self.epochs):
            for xi, target in zip(t_data, targets):
                update = self.lr * (target - self.predict(xi)) #Calculates error times lr for every sample iteratively
                self.weights_[1:] += update * xi #Accounts for multiplying by xi before incrementing the weights
                self.weights_[0] += update #Since x0 would equal 1, updates bias without needing x0 coefficient
            self.log_weights(self.weights_) # Logs weights to weights.csv
        return self
            

    
def csv_to_numpy(file, titanic):
    
    if titanic:
        dtype = [                           #Creates a structured numpy array to facilitate pre-processing requirements
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
        
        return np.genfromtxt(file, delimiter=',', skip_header=1, dtype=dtype, missing_values='', filling_values=np.nan)  # Generates a numpy array from a csv file, replacing empty string values w/ nan values
    else:
        return np.genfromtxt(file, delimiter=',', skip_header=1) # Cretes unstructured numpy array for datasets with consistent data types across all features
    


def get_column_numbers(prompt):  #Simple method to accept multiple arguments from the command line by splitting the tokens and inserting them into a list
    
    user_input = input(prompt) 
    if user_input.strip() == "":
        return []
    else:
        return [int(col) - 1 for col in user_input.split()]
    
def process_data(t_data, titanic):    # Core method for pre-processing data from both structured and unstructured datasets
    
    if titanic:

        target_col = input("\n\nEnter the column header name of the target values w/ proper punctuation (Press Enter for default): ")
        if target_col == '':
            target_col = 'Survived' #default exclusion
        

        unique_id_col = input("\n\nEnter the column header name of the unique IDs w/ proper punctuation (Press Enter for default): ")
        if unique_id_col == '':
            unique_id_col = 'PassengerId' # Default exclusion


        irrelevant_cols = get_column_numbers("\n\nEnter column header names of fields you want to exclude from training seperated by spaces w/ proper punctuation (Press Enter for default): ")
        if not irrelevant_cols:
            irrelevant_cols = ['LastName', 'FirstName', 'Ticket', 'Cabin'] # Default exclusion

        features = [col for col in t_data.dtype.names if col not in irrelevant_cols + [target_col, unique_id_col]]  # Creates a list of columns that should be included as training data columns/attributes

        X = np.column_stack([t_data[field] for field in features]) #Extracts column data specified by features and stacks them into a 2D array
        
        if X.ndim == 1:
            X = X.reshape(-1, 7) # Failsafe in case X is somehow still one-dimensional after previous statement (not sure why this happens sometimes)
        Y = np.asarray(t_data[target_col])  # Takes the target column data and stores it as a 1D numpy array/vector
        X = conversions(X) # Conducts conversions on structured arrays to map numerical values to categorical string variables
        X = standardize(X) # Standardizes all of the data fields that are intended to be included in the learning process


        
    else:

        target_col = int(input("\n\nEnter the column number (starting at 1) of the target values: ")) - 1   

        unique_id_col = int(input("\n\nEnter the column number (starting at 1) of the unique IDs: ")) - 1

        irrelevant_cols = get_column_numbers("\n\nEnter column numbers (starting at 1) to exclude from the training dataset seperated by spaces, or press Enter for none: ")

        irrelevant_cols = [col - 1 for col in irrelevant_cols]  # List iteration to obtain the index values of the user-specified irrelevant columns

        remove_columns = set([target_col, unique_id_col] + irrelevant_cols)  # Merges the unneeded columns for the training field dataset by merging the lists into a set

        X = np.delete(t_data, list(remove_columns), axis=1)  # Deletes all of the columns to be removed from the training field dataset
        Y = t_data[:, target_col] # Assigns Y as a vector holding the the training data target values
        

    return X, Y, target_col, unique_id_col, irrelevant_cols  #Returns several values to avoid having to request additional user input in the future

def conversions(X):
    
    X[:, 1] = np.where(X[:, 1] == 'male', 1, np.where(X[:, 1] == 'female', 2, np.nan))  # Convert 'Sex' from 'male', 'female' to 1, 2

    
    X[:, 0] = np.where(X[:, 0] == 3, 1, np.where(X[:, 0] == 2, 2, 3))  # Rotates 'Pclass' values to better match the positive linear relationship between 'Pclass' & 'Survived'
    
    
    valid_embarked = ['C', 'Q', 'S']
    X[:, 6] = np.where(np.isin(X[:, 6], valid_embarked), X[:, 6], 'Q')  # Handle 'Embarked' by replacing the missing or invalid values with 'Q'

    c = 3.0
    q = 2.0
    s = 1.0
    default = 2.0    # Used to specify float values for 'Embarked' replacement

    
    X[:, 6] = np.where(X[:, 6] == 'C', c, np.where(X[:, 6] == 'Q', q, np.where(X[:, 6] == 'S', s, default)))  # Converts 'Embarked' values to mapped numeric values

    X = X.astype(np.float64) # Converts training field dataset from unstructured string array to unstructured float array

    mean_age = np.nanmean(X[:, 2])
    X[:, 2] = np.where(np.isnan(X[:, 2]), mean_age, X[:, 2]) # Assigns missing values the average age of the entire training dataset

    return X


def standardize(X):
    
    for i in range(7):
        col_min = np.nanmin(X[:, i])
        col_max = np.nanmax(X[:, i])     # Finds the min & max value of every column of the training field dataset

    if col_min != col_max:
        X[:, i] = (X[:, i] - col_min) / (col_max - col_min)  # If the min & max aren't equal, standardizes the entire column using the max-min formula

    else:
        X[:, i] = 0   # If the max & min are equal, sets the entire column to zero to signal that the data dosen't provide any actionable action by the ML model 
    
    return X

def run_program():
    print("\n\n************** COMP 379 HW2 | Lueders, Sebastian **************\n\n\n")
    
    
    td_file = input("What is the name of the csv file you'd like to use for training (Must be located in program's directory)?: ")


    if td_file == 'titanic_train.csv' or td_file == 'titanic_test.csv':
        titanic = True
        t_data = csv_to_numpy(td_file, titanic)
    else:
        titanic = False
        t_data = csv_to_numpy(td_file, titanic)     # Determines if the titanic dataset is being used so that we can anticipate using a structured array in the future


    TX, TY, target_col, unique_id_col, irrelevant_cols = process_data(t_data, titanic)  # Conducts data pre-processing

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
        model.set_is_adaline(False)     # Initializes ML Model & sets the corresponding instance variable to indicate the proper ML Model steps to take
    
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
        print(f"\n\n*** Accuracy Rate: {accuracy * 100:.2f}%")   # Prepares testing data in a similar manner to the training data before testing
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
            print(f"\n\n*** Accuracy Rate: {accuracy * 100:.2f}%")       # Prepares testing data in a similar manner to the training data before testing
    
    
    
    record_output = input("\n\nWould you like to save the model output in a seperate file?(y/n): ").lower().strip()

    if record_output == 'n':
        pass
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
    
        print(f"\n\nResults saved to {ofile_name}")   # Offers an option to export results to a file of user's choosing

    bt = input("\n\nDo you want to run a baseline test? (y/n): ")
    if bt == 'y':
        t_data = csv_to_numpy("titanic_train.csv", titanic=True)
        p_data = csv_to_numpy("titanic_test.csv", titanic=True)
        BX1, BY1, target_col, unique_id_col, irrelevant_cols = process_data(t_data, titanic)
        BX2, BY2, target_col, unique_id_col, irrelevant_cols = process_data(p_data, titanic)
        baseline_model = LinearModelBase()
        baseline_model.initialize_weights(BX1)
        base_prediction1 = baseline_model.predict(BX1, baseline=True)
        base_prediction2 = baseline_model.predict(BX2, baseline=True)
        accuracy1 = baseline_model.check_accuracy(base_prediction1, BY1)
        accuracy2 = baseline_model.check_accuracy(base_prediction2, BY2)
        print(f"\n\n*** Accuracy Rate on Training Data: {accuracy1 * 100:.2f}%") 
        print(f"\n\n*** Accuracy Rate on Test Data: {accuracy2 * 100:.2f}%\n\n") 

    else:
        print("\n\nGoodbye!\n\n")

    


class ModelPlots:
    def __init__(self, t_file='train.csv'):
        self.td = pd.read_csv(t_file)

    def quick_scatter(self, x_col, y_col, title="", xlabel="", ylabel="", color='black', size=50, annotate=False, folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.td[x_col], self.td[y_col], s=size, c=color, alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        if annotate:
            for i in range(len(self.td)):
                plt.text(self.td[x_col].iloc[i], self.td[y_col].iloc[i], 
                         f"({self.td[x_col].iloc[i]:.2f}, {self.td[y_col].iloc[i]:.2f})",
                         fontsize=9, ha='right')
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Scatter.png'))
        plt.close()

    def quick_box(self, x_col, y_col, title="", xlabel="", ylabel="", width=10, height=6, bin_size=10, folder_name="Graphics"):
        plt.figure(figsize=(width, height))
        
        self.td['Binned_' + x_col] = pd.cut(self.td[x_col], bins=range(0, int(self.td[x_col].max()) + bin_size, bin_size), right=False)
        
        self.td.boxplot(column=y_col, by='Binned_' + x_col, grid=False, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red'),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'),
                        flierprops=dict(marker='o', color='red', alpha=0.5))
        
        plt.title(title, fontsize=18, weight='bold')
        plt.suptitle('')  
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Box.png'))
        plt.close()

    def quick_bar(self, x_col, y_col, title="", xlabel="", ylabel="", folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        means = self.td.groupby(x_col)[y_col].mean()
        means.plot(kind='bar', color='lightblue', edgecolor='blue')
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Bar.png'))
        plt.close()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_regression_curve(self, X_col, y_col, title="", xlabel="", ylabel="Probability", folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        X = self.td[X_col]
        y = self.td[y_col]
        X_norm = (X - X.mean()) / X.std()
        X_test = np.linspace(X_norm.min(), X_norm.max(), 300)
        y_prob = self.sigmoid(X_test)
        plt.plot(X, self.sigmoid(X_norm), color='blue', lw=2, label='Logistic Curve')
        plt.scatter(X, y, color='red', edgecolor='k', s=100, label='Data Points')
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{X_col}_Sig.png'))
        plt.close()

    def hist(self, X_col, bins, title="", xlabel="", ylabel="", folder_name="Graphics"):
        x = self.td[X_col]

        plt.hist(x, bins, edgecolor='blue', alpha=0.6, color='lightblue')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.grid(True)

        plt.show()
        plt.savefig(os.path.join(folder_name, f'{X_col}_Hist.png'))
        plt.close()



    def plot():
        while True:
            try:
                f_name = input("\nWhat would you like to name the folder to store the graphics? (Leave blank for the default 'Graphics' folder)\n> ")

                if f_name == '':
                    folder_name = "Graphics"
                else:
                    folder_name = f_name

                if os.path.exists(folder_name):
                    overwrite = input(f"The folder '{folder_name}' already exists. Do you want to overwrite it? (y/n): ").lower()
                    if overwrite == 'y':
                        shutil.rmtree(folder_name)  
                        os.makedirs(folder_name)  
                        print(f"\nFolder '{folder_name}' has been overwritten.\n\n")
                    else:
                        print("Please provide a different folder name.\n\n")
                        continue
                else:
                    os.makedirs(folder_name) 
                    print(f"Folder '{folder_name}' created successfully.\n\n")
                break 

            except Exception as e:
                print(f"An error occurred: {e}. Please try again.\n\n")


if __name__ == "__main__":
    
    while True:
        create_graphic = input("Would you like to view and export data visualization materials? (y/n): ").lower().strip()

        if create_graphic == 'y':
            inst = ModelPlots() 

            folder_name = "Graphics"

            inst.hist(X_col='Fare', bins = 50, title="Fare Distribution of Titanic Passengers", xlabel="Fare", ylabel="Frequency", folder_name=folder_name)

            inst.quick_scatter(x_col='Age', y_col='Survived', title="Survival Rate by Passenger's Age", xlabel="Age", ylabel="Survival", color='blue', folder_name=folder_name)

            inst.quick_scatter(x_col='Fare', y_col='Survived', title="Survival Rate by Passenger's Fare", xlabel="Fare", ylabel="Survival", color='blue', folder_name=folder_name)

            inst.logistic_regression_curve(X_col='Age', y_col='Survived', title="Survival Probability by Passenger's Age", xlabel="Age", ylabel="Probability of Survival", folder_name=folder_name)

            inst.logistic_regression_curve(X_col='Fare', y_col='Survived', title="Survival Probability by Passenger's Fare", xlabel="Fare", ylabel="Probability of Survival", folder_name=folder_name)

            inst.quick_box(x_col='Age', y_col='Survived', title="Distribution of Survival Rate by Passenger's Age", xlabel="Age", ylabel="Survival", bin_size=10, folder_name=folder_name) 

            inst.quick_box(x_col='Fare', y_col='Survived', title="Distribution of Survival Rate by Passenger's Fare", xlabel="Fare", ylabel="Survival", width=7, height=6, bin_size=50, folder_name=folder_name) 

            inst.quick_bar(x_col='Pclass', y_col='Survived', title="Distribution of Survival Rate by Passenger's Class", xlabel="Passenger Class", ylabel="Average Survival Rate", folder_name=folder_name) 

            inst.quick_bar(x_col='Sex', y_col='Survived', title="Distribution of Survival Rate by Passenger's Sex", xlabel="Sex", ylabel="Average Survival Rate", folder_name=folder_name)

            inst.quick_bar(x_col='Embarked', y_col='Survived', title="Distribution of Survival Rate by Passenger's Departure Location", xlabel="Port Embarked", ylabel="Average Survival Rate", folder_name=folder_name)

            inst.quick_bar(x_col='SibSp', y_col='Survived', title="Survival Rate by Number of Siblings & Spouses Onboard", xlabel="Number of Siblings/Spouses Onboard", ylabel="Average Survival Rate", folder_name=folder_name)

            inst.quick_bar(x_col='Parch', y_col='Survived', title="Survival Rate by Number of Parents & Children Onboard", xlabel="Number of Parents/Children Onboard", ylabel="Average Survival Rate", folder_name=folder_name)

            break
        elif create_graphic == 'n':
            break
        else:
            print("\nNot a valid response, please try again...\n\n")

    run_program() # Run main program for building Adaline & Perceptron Models


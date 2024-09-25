# AdalineML

The Adaline/Perceptron Model Template was Sourced from Python Machine Learning by Sebastian Raschka 

<b>COMMAND TO RUN:</b> python3 LinearModelBase.py

<b>Dependencies:</b>
1. python3
2. os
3. shutil
4. numpy 
5. csv
6. shutil
7. matplotlib
8. pandas


<b>Limitations:</b>
*** This program will only handle custom datsets with every column representing numeric data
*** For the mixed data type dataset, this program is only built to accept titanic datasets with the following headers:
PassengerId,Survived,Pclass,LastName,FirstName,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

<b>Repository Information:</b>

LinearModelBase.py -> Primary algorithm; allows for users to train and test Perceptron & (Batch Gradient) Adaline Models on custom csv datasets & provides users with graphing capabilities for the train.csv dataset (no CL customization on graphing yet!)


SummitSuccess.csv -> Linearly-seperated, synthetic dataset used to investigate algorithm behavior


SoccerGoalData.csv -> Non-linearly seperated, synthetic dataset used to demonstrate the pitfalls of linear ML models like Perceptron & Adaline


weights.csv -> Logs weights after each epoch; can be used to test for convergence with the perceptron model


train.csv -> The original training dataset imported from Kaggle with no edits


titanic_train.csv -> Random-batch split of 70% of the original titanic training dataset


titanic_test.csv -> Random-batch split of 30% of the original titanic training dataset


adaline-on-test-data-results.csv -> Output of the results of the adaline model trained on titanic_train.csv & tested on titanic_test.csv


adaline-on-training-data-results.csv -> Output of the results of the adaline model trained & tested on titanic_train.csv


perceptron-on-test-data-results.csv -> Output of the results of the perceptron model trained on titanic_train.csv & tested on titanic_test.csv


perceptron-on-training-data-results.csv -> Output of the results of the perceptron model trained & tested on titanic_train.csv


perceptron-on-SoccerGoalData.csv -> Output of the results of the perceptron model trained & tested on SoccerGoalData.csv


perceptron-on-SummitSuccess.csv -> Output of the results of the perceptron model trained & tested on SummitSuccess.csv


Graphics -> Folder containing the exported graphics sourced from accepting the graph option while running the main program








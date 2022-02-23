#Kenneth Chuson

'''
 TODO:
     Step 1) Create Training set & Test set 

        1.a) Create a toy labeled dataset for a two-class problem (e.g. spam vs ham OR positive vs negative).  This dataset should have 20 datapoints (sentences) with a class label associated with each datapoint.  

        1.b) Split this dataset into Training set (80%) and Test set (20%). 

        1.c) Store the created data into two files: train.csv and test.csv
        
    Step 2) Train a Naive Bayes classifier

        2.a) Read in the train.csv file 

        2.b) Calculate the prior probabilities for both classes using the training data

        2.b) Calculate the multinomial distribution for each class, i.e. the conditional probability of each word given the class using the training data

        2.c) Output the learnt classification model (all the above probabilities) to a file named model.csv

    
    Step 3) Use the learnt NB classifier

        .a) Read in the model.csv file and the test.csv files 

        3.b) Use the classification model to predict class label for each datapoint in the test set

        3.c) Output the test datapoints (sentences) and their predicted labels to a file named test_predictions.csv
'''
import csv

class Text_Classification_Using_Naive_Bayes(object):

    def __init__(self, toy_labeled_dataset):
        self.toy_labeled_dataset = toy_labeled_dataset
        self.store_text = [] 
    
    def Reading_file(self):
        with open(self.toy_labeled_dataset, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader:
                print(line[0])
                self.store_text.append(line[0])
                
    def calculate(self):
        print("------------calculate-----------") 

        for i in range(len(self.store_text)):
            print(self.store_text[i]) 
        
    

file_csv_name = "models.csv"

class_text_classify = Text_Classification_Using_Naive_Bayes(file_csv_name)

class_text_classify.Reading_file()


class_text_classify.calculate() 





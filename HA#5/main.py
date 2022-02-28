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
from __future__ import division
import csv


class Text_Classification_Using_Naive_Bayes(object):

    def __init__(self, toy_labeled_dataset):
        #train variables
        self.toy_labeled_dataset = toy_labeled_dataset
        self.store_label = [] 
        self.store_text = []
        self.assign_text_label = {} #key -> sentences : value -> labels
        self.map_label_one_text = {} #key -> sentences : value -> Ham class 
        self.map_label_two_text = {} #key -> sentences : value -> Spam class
        self.store_label_one = []
        self.store_label_two = []
        self.store_text_from_label_one = []
        self.store_text_from_label_two = []
        
    
    def Reading_train_file(self):
        with open(self.toy_labeled_dataset, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader:
                self.store_label.append(line[0])
                self.store_text.append(line[1])
                if line[0] == 'ham':
                    self.store_label_one.append(line[0])
                    self.store_text_from_label_one.append(line[1]) 
                if line[0] == 'spam':
                    self.store_label_two.append(line[0])
                    self.store_text_from_label_two.append(line[1]) 

        self.map_label_one_text = dict(zip(self.store_text_from_label_one, self.store_label_one))
        self.map_label_two_text = dict(zip(self.store_text_from_label_two, self.store_label_two))  
        
        self.assign_text_label = dict(zip(self.store_text, self.store_label))


                
    def calculate_training_sets(self):
        print("------------calculate-----------")



        #getting labels
        
        #ham
        for i in range(len(self.store_label_one)):
            #print(self.store_label_one[i])
            pass

        #spam 
        for i in range(len(self.store_label_two)): 
            #print(self.store_label_two[i]) 
            pass
        

        #getting sentences

        #ham
        print("HAM sentences train: ")
        for i in range(len(self.store_text_from_label_one)):
            #print(self.store_text_from_label_one[i])
            pass
        
        #spam 
        print("SPAM sentences train: ")
        for i in range(len(self.store_text_from_label_two)): 
            #print(self.store_text_from_label_two[i]) 
            pass
        
        prob_one = self.prior_Probabilties()[0] 
        prob_two = self.prior_Probabilties()[1] 
        
        #Prior Probabilities 
        print("Prior Probability Ham: ", prob_one)
        print("Prior Probability Spam: ", prob_two) 

        #keep track of each word 

        store_split_one = [] 
        store_split_two = [] 
        store_split_all_words = [] 

        #spam
        for i in range(len(self.store_text_from_label_one)):  
            splitting_convert_one = self.store_text_from_label_one[i].split(" ") 
            for j in range(len(splitting_convert_one)): 
                store_split_one.append(splitting_convert_one[j]) 
        
        #ham 
        for i in range(len(self.store_text_from_label_two)): 
            splitting_convert_two = self.store_text_from_label_two[i].split(" ") 
            for j in range(len(splitting_convert_two)): 
                store_split_two.append(splitting_convert_two[j])  


        #all words
        combine_arr = self.store_text_from_label_one + self.store_text_from_label_two 

        for i in range(len(combine_arr)):  
            splitting_convert_three = combine_arr[i].split(" ") 
            for j in range(len(splitting_convert_three)):
                store_split_all_words.append(splitting_convert_three[j])
        
        vocab_size = len(list(set(store_split_all_words)))



        def calculate_prob_label_one(word_chose): 
            dic_calc = {} 
            count = store_split_one.count(word_chose) 
            prob = (count + 1) / (len(store_split_one) + vocab_size)
            dic_calc[count] = prob 

            return [ i for i in dic_calc.values()]

    
        def calculate_prob_label_two(word_chose): 
            dic_calc = {} 

            count_2 = store_split_two.count(word_chose)
            prob_2 = (count_2 + 1) / (len(store_split_two) + vocab_size)  
            dic_calc[count_2] = prob_2 

            return [ i for i in dic_calc.values() ]
        
        '''
            dic_label ( key: (<spam or ham>, <word>) value: probability each word ) 
        
        '''
        dic_label_one = {} 
        dic_label_two = {} 

        for (key, val) in self.assign_text_label.items(): 
            if val == "ham":
                a1 = key.split(" ") 

                for word in a1: 
                    count_a1 = store_split_all_words.count(word) 
                    prob = (count_a1 + 1) / (len(store_split_one) + vocab_size) 
                    dic_label_one[(val, word)] = prob 


            if val == "spam": 
                a2 = key.split(" ") 
                
                for word in a2: 
                    count_a2 = store_split_all_words.count(word) 
                    prob = (count_a2 + 1) / (len(store_split_one) + vocab_size)
                    dic_label_two[(val, word)] = prob 

        print("label one: ", dic_label_one) 
        print("label two: ", dic_label_two) 

        
            
        


    def prior_Probabilties(self): 

        total_size = len(self.store_label_one + self.store_label_two)
        
        prior_probability_label_one = len(self.store_label_one) / total_size 
        prior_probability_label_two = len(self.store_text_from_label_one) / total_size

        return [prior_probability_label_one, prior_probability_label_two]
    
    
        




        
        
    


        print("---Naive Bayes Classifier---")

        
        '''

         Calculate the prior probabilities for both classes using the training data

         P(<#word> | <#number of words in class_one or class_two>) = count(<#word>) / count(<#number of words in class_one or class_two>) 

        '''
        
        
        
        
    

file_csv_name = "train.csv"

class_text_classify = Text_Classification_Using_Naive_Bayes(file_csv_name)

class_text_classify.Reading_train_file()


class_text_classify.calculate_training_sets() 





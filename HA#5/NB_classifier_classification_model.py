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

    def __init__(self, toy_labeled_dataset_train, toy_labeled_dataset_test, toy_labeled_dataset_model,  toy_labeled_dataset_test_predictions):

        '''
        These stores will store the words, sentences, labels (either ham or spam) to seperate and organize these into categories. 
        
        
        '''
 
        self.toy_labeled_dataset_train = toy_labeled_dataset_train
        self.toy_labeled_dataset_test = toy_labeled_dataset_test 
        self.toy_labeled_dataset_model = toy_labeled_dataset_model
        self.toy_labeled_dataset_test_predictions = toy_labeled_dataset_test_predictions
        self.store_label = [] 
        self.store_text = []
        self.assign_text_label = {} #key -> sentences : value -> labels
        self.map_label_one_text = {} #key -> sentences : value -> Ham class 
        self.map_label_two_text = {} #key -> sentences : value -> Spam class
        self.store_label_one = []
        self.store_label_two = []
        self.store_text_from_label_one = []
        self.store_text_from_label_two = []

        '''
        These stores are for test and same thing to seperate and organize thsese into categories. 
        
        '''

        self.store_label_test = [] 
        self.store_text_test = [] 
        self.store_label_one_test = [] 
        self.store_text_from_label_one_test = [] 
        self.store_text_from_label_two_test = [] 
        self.store_label_two_test = [] 
        self.store_label_from_label_two_test = [] 
        self.map_label_one_text_test = {} 
        self.map_label_two_text_test = {} 
        self.assign_text_label_test = {} 

        
    
    def Reading_train_file(self):
        '''
            while reading the csv train file. Let store for text and labels. Also combine them together and which sentences belong 
            to specific labels using a dictionary. 
        '''
        with open(self.toy_labeled_dataset_train, 'r') as csv_file:
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


    
    def Reading_test_file(self): 
        '''
            while reading the csv test file. Let store for text and labels. Also combine them together and which sentences belong 
            to specific labels using a dictionary. 
            After that, it calls to the calculating the training sets function. 
        '''
        with open(self.toy_labeled_dataset_test, 'r') as csv_file: 
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader: 
                self.store_label_test.append(line[0]) 
                self.store_text_test.append(line[1]) 
                if line[0] == 'ham': 
                    self.store_label_one_test.append(line[0]) 
                    self.store_text_from_label_one_test.append(line[1]) 
                if line[0] == 'spam': 
                    self.store_label_two_test.append(line[0]) 
                    self.store_text_from_label_two_test.append(line[1]) 

        self.map_label_one_text_test = dict(zip(self.store_text_from_label_one_test, self.store_label_one_test))
        self.map_label_two_text_test = dict(zip(self.store_text_from_label_two_test, self.store_label_two_test))  
        
        self.assign_text_label_test = dict(zip(self.store_text_test, self.store_label_test))


        self.calculate_training_sets(self.assign_text_label_test)



        

    def Write_model_file(self, store_label_one_label, store_label_one_word, store_probability_one_word, store_label_two_label, store_label_two_word, store_probability_two_word): 

        '''
        Writing to a model csv to store each word, labels, and probability of each word. 
        '''


        with open(self.toy_labeled_dataset_model, 'w') as csv_file: 
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

            for (x, y, z) in zip(store_label_one_label, store_label_one_word, store_probability_one_word): 
                writer.writerow([x, y, z])

            for (x, y, z) in zip(store_label_two_label, store_label_two_word, store_probability_two_word): 
                writer.writerow([x, y, z])
    

    def Write_test_prediction_file(self, sentences_test_trained, probabilities_test_trained): 
        with open(self.toy_labeled_dataset_test_predictions, 'w') as csv_file: 
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

            for (x, y) in zip(sentences_test_trained, probabilities_test_trained): 
                writer.writerow([x, y])
            
            


                
    def calculate_training_sets(self, test_chose):

        #the test_chose is from the test sentences and labels. 

        print("------------Calculate Naive Bayes Classifier-----------")

        print("--------NB Classifier-------") 

        
        prob_one = self.prior_Probabilties()[0] 
        prob_two = self.prior_Probabilties()[1] 
        
        #Prior Probabilities 
        print("Prior Probability Ham: ", prob_one)
        print("Prior Probability Spam: ", prob_two) 

        #keep track of each word 

        '''
            Reason using split because we want to split the sentences into each word, so it is easy to calculate and keep of track of words in each sentence. 
        
        '''

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


       

        
        '''
        
            dic_label ( key: (<spam or ham>, <word>) value: probability each word ) 

            Using dic_label to organize and keep track of categories between labels, words, and probability of each word. 
        
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

        '''
        
        Print out the each word of their probability. 

        Label one is ham. 
        Label two is spam. 
        
        '''

        print("label one: ", dic_label_one) 
        print("label two: ", dic_label_two) 


        '''
        Using separate arrays to store specific labels, probability and words. So that it is easier to handle while writing to a model.csv
        
        '''

        store_label_one_label = [] 
        store_label_one_word = [] 
        store_probability_one_word = [] 

        store_label_two_label = [] 
        store_label_two_word = [] 
        store_probability_two_word = [] 


       


    
        for (key, val) in dic_label_one.items(): 
            label_one_label = key[0]
            label_one_word = key[1]
            probability_one_word = val 
            store_label_one_label.append(label_one_label) 
            store_label_one_word.append(label_one_word)
            store_probability_one_word.append(probability_one_word)
            
        
        for (key, val) in dic_label_two.items(): 
            label_two_label = key[0]
            label_two_word = key[1]
            probability_two_word = val 
            store_label_two_label.append(label_two_label)
            store_label_two_word.append(label_two_word)
            store_probability_two_word.append(probability_two_word)
        


        self.Write_model_file(store_label_one_label, store_label_one_word, store_probability_one_word, store_label_two_label, store_label_two_word,   store_probability_two_word)


        '''

        These two functions have the same computation. Except the size of two classes (ham and spam) 
        
        
        '''

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

        This is for the test to caluclate the predicting class labels and write into test_predictions.csv file. 
        
        
        '''

        print("--------NB Classification model-------") 



        prod_res_class_one = 1
        prod_res_class_two = 1 

        prediction_test_result = {} 

        for (key, val) in test_chose.items(): 
            split_text_test = key.split(" ")


            if (val == "ham"): 

                for word in split_text_test: 
                    out1 = calculate_prob_label_one(word) 
                    prod_res_class_one *= out1[0] 
                    prediction_test_result[key] = prob_one * prod_res_class_one

            if (val == "spam"): 

                for word in split_text_test:   
                    out2 = calculate_prob_label_two(word)
                    prod_res_class_two *= out2[0]
                    prediction_test_result[key] = prob_two * prod_res_class_two
            

        res_class_one_chose = prob_one * prod_res_class_one
        res_class_two_chose = prob_two * prod_res_class_two

        print("Predict class one (ham): ", res_class_one_chose)
        print("Predict class two (spam): ", res_class_two_chose)


        sentences_test_trained = [] 
        probabilities_test_trained = [] 

        for (key, val) in prediction_test_result.items(): 
            sentences_test_trained.append(key) 
            probabilities_test_trained.append(val) 
        

        self.Write_test_prediction_file(sentences_test_trained, probabilities_test_trained)






    

    def prior_Probabilties(self): 

        '''
            Calculating the prior probabilities
        '''

        total_size = len(self.store_label_one + self.store_label_two)
        
        prior_probability_label_one = len(self.store_label_one) / total_size 
        prior_probability_label_two = len(self.store_text_from_label_one) / total_size

        return [prior_probability_label_one, prior_probability_label_two]
    

    
    

        
        
        
        
    

file_csv_names = ["train.csv", "test.csv", "models.csv", "test_predictions.csv"]


class_text_classify = Text_Classification_Using_Naive_Bayes(file_csv_names[0], file_csv_names[1], file_csv_names[2], file_csv_names[3])

class_text_classify.Reading_train_file()

class_text_classify.Reading_test_file() 






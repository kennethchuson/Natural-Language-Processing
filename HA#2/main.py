

import nltk
from nltk.tokenize import word_tokenize

nltk.download('gutenberg')


class SimpleWordAutoComplete(object):

    def __init__(self, input_argument, what_textFile):
        self.input_argument = input_argument
        self.what_textFile = what_textFile
        self.tokenize_input_argument = word_tokenize(self.input_argument) 

    def getListWords(self):
        return self.tokenize_input_argument

    def printFiles_from_gutenberg(self):
        return nltk.corpus.gutenberg.fileids()

    def func_WordAutocomplete(self):
        pass

        #1) Build a vocabulary (set of all unique words) using any English corpus from nltk.
        
        get_list_gutenberg_text = nltk.corpus.gutenberg.words(self.what_textFile)

        starting_word_number = 0
        ending_word_number = 500
        number_occurrences = {}
        store_words = []
        check_if_exist_from_your_input = False
        levenshtein_distance_store = {} 


        print("words in a ", what_textFile) 
        for word in get_list_gutenberg_text[starting_word_number:ending_word_number]: 
            print(word, sep=' ', end=' ')
            store_words.append(word) 

        
        #2) Find the number of occurrences (frequency) of each word in the vocabulary.  Also, find the total number of words in the chosen corpus (N).
        print("number of occurences (frequency) of each word in a vocabulary")
        
        for word in get_list_gutenberg_text[starting_word_number:ending_word_number]:
            if word not in number_occurrences:
                number_occurrences[word] = 0
            number_occurrences[word] += 1
            print("Number occurences of ", number_occurrences.keys(), " is ", number_occurrences.values())

        print("total number of words", len(store_words))

        #3) Find the relative frequency of each word W where relative frequency of W = frequency_of_W / N. This relative frequency can be interpreted as the probability (likelihood) of each word in the corpus.
        for word in get_list_gutenberg_text[starting_word_number:ending_word_number]:
            if word not in number_occurrences:
                number_occurrences[word] = 0
            number_occurrences[word] += (1 / len(store_words)) 
            print("Relative Frequency of ", number_occurrences.keys(), " is ", number_occurrences.values())

        #4) 

        #4.a) If the input string XYZ exists in your vocabulary, return "XYZ is a complete and correct word in English."
            

        for word in get_list_gutenberg_text[starting_word_number:ending_word_number]:
            if self.input_argument in word:
                check_if_exist_from_your_input = True

        if check_if_exist_from_your_input:
            print(self.input_argument, " is a complete and correct word in English.")
            

        #4.b) If the input string doesn't exist in your vocabulary, perform the below steps:

        if check_if_exist_from_your_input == False:
            
            #4.b.i) Calculate the similarity between each word in the vocabulary and the input string using Levenshtein distance. (Use any open-source python library for calculating Levenshtein distance.)
            for word in get_list_gutenberg_text[starting_word_number:ending_word_number]: 
                levenshetin_distance = self.LevenshteinDistance_func(self.input_argument, word) 
                print(word, " leveinshtein distance is ", levenshetin_distance)
                levenshtein_distance_store[levenshetin_distance] = word 
            
            print("top words")
            for i in sorted(levenshtein_distance_store): 
                a = (i, levenshtein_distance_store[i])
                print(a) 
            


    
    def LevenshteinDistance_func(self, word1, word2):
    
        m = len(word1)
        n = len(word2)
        table = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            table[i][0] = i
        for j in range(n + 1):
            table[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
                    
        return table[-1][-1]
            
            

        
             



        
        
            


input_argument = str(input("input your string: "))

what_textFile = 'shakespeare-hamlet.txt'

word_autoComplete = SimpleWordAutoComplete(input_argument, what_textFile)

print(word_autoComplete.getListWords())
#print(word_autoComplete.func_WordAutocomplete())
word_autoComplete.func_WordAutocomplete()
#Kenneth Chuson



'''
    Todo:
        Goal:
        1.) Let the user input a word
        2.) then should output the list of words that can follow the input word
        3.) And corresponding probailities

        Requirements to build:
         * By building a bigram LM
            1.) Use NLTK library
            2.) Compute probability of each bigram using MLE (count(w1 w2)/count(w1))
         * Predict next word using the following steps:
            1.) Get an input word from user, inpW.
            2.) Use the bigram LM built in step 1 to find all the bigrams where the input word, inpW, is w1.
                Display all possible next words from these bigrams and their corresponding probabilities.
                (Sort in descending order on probabilities)
        
Example:
    from_nltk_toy_corpus = "I will go to California to meet my friend"

    Output of Step 1: Biagram model:
                            P(will | I) = 1
                            P(go | will) = 1
                            P(to | go) = 1
                            P(California | to) = 0.5
                            P(to | California) = 1
                            P(to | California) = 1
                            P(meet | to) = 0.5
                            P(my | meet) = 1
                            P(friend | my) = 1
     user_input = "to"

     Output: "Possible next words: California: 0.5, meet: 0.5"
'''

import nltk
from operator import itemgetter


nltk.download('gutenberg')

class Next_word_predictor_using_a_bigram_language_model(object):

    def __init__(self, user_input, what_textFile):
        self.user_input = user_input
        self.what_textFile = what_textFile
        self.get_list_gutenberg_text = nltk.corpus.gutenberg.words(what_textFile)
        self.store_text = []
        self.bigram_text = []


    def print_words_from_nltk_corups(self):
        starting_word_number = 0
        ending_word_number = 1000

        print("words from ", what_textFile) 
        for word in self.get_list_gutenberg_text[starting_word_number:ending_word_number]:
            self.store_text.append(word)
            print(word, sep='', end='\n')

    def print_bigrams_words(self):
        
        self.bigram_text = list(nltk.bigrams(self.store_text))

    def probability_bigrams_words(self):
        pass
        '''
          w1 = frequency for w1
          w2 = frequency for w2
        
          prob = (count(w1, w2)/count(w1))
        '''
        sentence_text = ' '.join(word for word in self.store_text)
        
   
        store_count_w1 = []
        map_bigrams_words = {}
        result = []
        result_map = {} 
        
        for i in range(len(self.store_text)):
            store_count_w1.append(sentence_text.count(self.store_text[i]))

        for (word1, word2) in self.bigram_text:
            if (word1, word2) in map_bigrams_words:
                map_bigrams_words[(word1, word2)] += 1
            else:
                map_bigrams_words[(word1, word2)] = 1
        
        for (num), (key, value) in zip(store_count_w1, map_bigrams_words.items()):
            calculate = value / num
            if self.user_input in key:
                result_map[key] = calculate

        K = 2

        result = dict(sorted(result_map.items(), key = itemgetter(1))[:K])

        print("possible next words: ")
        for (key, value) in result.items(): 
            print(key[1] ,":", value)
    
        
    


user_input = str(input("Input a word: "))
what_textFile = 'shakespeare-hamlet.txt'


next_word_predict = Next_word_predictor_using_a_bigram_language_model(user_input, what_textFile)

next_word_predict.print_words_from_nltk_corups()

next_word_predict.print_bigrams_words()

next_word_predict.probability_bigrams_words() 





                            
                            


            
            
        
    

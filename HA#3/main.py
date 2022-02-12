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
        '''
            user_input - takes the user input required.
            what_textFile - what the textfile is.
            get_list_gutenberg_text - this will give the list of words from the textfile from the gutenberg corpus nltk library.
            store_text - store the list of texts.
            bigram_text - store the list of bigram text. 
        '''
        self.user_input = user_input
        self.what_textFile = what_textFile
        self.get_list_gutenberg_text = nltk.corpus.gutenberg.words(what_textFile)
        self.store_text = []
        self.bigram_text = []


    def print_words_from_nltk_corups(self):
        '''
            starting_word_number - is set to 0. 
            ending_word_number - is set to 1000 words for now.

            self.store_text.append(word) - this will store each of the text from the corpus. 
        '''
        starting_word_number = 0
        ending_word_number = 1000

        print("words from ", what_textFile) 
        for word in self.get_list_gutenberg_text[starting_word_number:ending_word_number]:
            self.store_text.append(word)
            print(word, sep='', end='\n')

    def print_bigrams_words(self):
        '''
            bigram_text - get the bigram text and this will get the list of tuples in an array.

            [(word1, word2), (word2, word3), (word3, word4) .... (word(n - 1), word(n))] 
    
        '''
        self.bigram_text = list(nltk.bigrams(self.store_text))

    def probability_bigrams_words(self):
        pass
        '''
        Goal: 
              w1 = frequency for w1
              w2 = frequency for w2
            
              prob = (count(w1, w2)/count(w1))

        sentence_text - this will convert words in an array from the store_text.

        store_count_w1 - keep on track how many words in them.
                    given_sentence = "I will go to California to meet my friend"
                    For example: ["I", "will", "go", "to", "California", "to", "meet", "my", "friend"]
                        this will convert to [1, 1, 1, 2, 1, 2, 1, 1, 1] for the each word frequencies count.

        map_bigrams_words - This will map the list of tuples bigram words
                                                    (word1, word2)                frequencies as number (n) 
                                map_bigrams_words[<Where the tuples as keys>] = <then the frequencies>
        result - store the result as dictionary of map_bigrams_words after computing the probability (prob).

        result_map - this will help to print out the result. 
         
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

        '''
            K - how many times do you want to see the closer probability from the user input.

            result - using operator python library to help determine sorting dictionary.
            
        '''

        K = 3

        result = dict(sorted(result_map.items(), key = itemgetter(1))[:K])

        print("possible next words: ")
        for (key, value) in result.items(): 
            print(key[1] ,":", value)
    
        
    
'''
    user_input - takes the user input of the word.
    what_textFile - determine what the text file is. For right now "shakespear-hamlet.txt" is the
                    default file for now.
    next_word_predict - call a class and this class has core required functions of this assignment.

    print_words_from_nltk_corups - it will print the words from the nltk corpus.

    print_bigrams_words - orint the words with the lists of bigram words.

    probability_bigrams_words - call the compute probability of bigrams words and analyze from the user input. 


'''

user_input = str(input("Input a word: "))
what_textFile = 'shakespeare-hamlet.txt'


next_word_predict = Next_word_predictor_using_a_bigram_language_model(user_input, what_textFile)

next_word_predict.print_words_from_nltk_corups()

next_word_predict.print_bigrams_words()

next_word_predict.probability_bigrams_words() 





                            
                            


            
            
        
    

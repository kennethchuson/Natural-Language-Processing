

import nltk
from nltk.corpus import words

nltk.download('words')


class SimpleWordAutoComplete(object):

    def __init__(self, input_argument):
        self.input_argument = input_argument
        self.String_Characters = self.input_argument.split()

    def getListWords(self):
        return self.String_Characters

    def func_WordAutocomplete(self):
        your_word = set(self.String_Characters)
        check_exist = False
        total_number_words = 0
        number_occurance = 0 
        
        for word in words.words(): 
            if word in your_word:
             total_number_words += 1
             check_exist = True
             
        return { "exist words": check_exist,
                 "total number words": total_number_words
               }
            

    


input_argument = str(input("input your string: "))

word_autoComplete = SimpleWordAutoComplete(input_argument)

print(word_autoComplete.getListWords())
print(word_autoComplete.func_WordAutocomplete()) 

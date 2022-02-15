#Kenneth Chuson

'''
    TODO:
        Language generation using n-grams

        1.) Analyzing: 
            https://www.numpyninja.com/post/n-gram-and-its-use-in-text-generation
'''



'''

    Meaning:
    pad_sequence - when you have a sequence (array of nums 'data'), it determines depending on the degree of the ngrams
                   Then, add the pads like <s> and <s/> 

    bigrams - gives you the list of 2 size evrey tuples (word1, word2) in an array

    ngrams - gives you the n-models such as unigram, bigrams, thrigrams and so fort. 

    everygrams - gives you all possible combinations of splitting tuples and also you can set to whatever max size of n-grams you want.

    pad_both_ends - will determine the pads such as <s> or/and </s>

    flatten - flattened (populate) from the given pad_sequence. 


    
    Documentation libraries:
            pad_sequence
        1.) https://tedboy.github.io/nlps/generated/generated/nltk.pad_sequence.html
        
            bigrams
        2.) https://www.tutorialspoint.com/python_text_processing/python_bigrams.htm

            ngrams
        3.) https://www.askpython.com/python/examples/n-grams-python-nltk

            everygrams
        4.) https://tedboy.github.io/nlps/generated/generated/nltk.everygrams.html

            pad_both_ends
        5.) https://www.nltk.org/_modules/nltk/lm/preprocessing.html
        
            flatten
        6.) https://www.nltk.org/api/nltk.lm.html
        

'''


from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten

'''

    these libraries will modify and read the trump's tweets from the csv using Kaggle. 
'''
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer

'''
    Define 2 different sentences into a matrix. 

    Determine bigrams and ngrams

    

'''

text = [
        ['I','need','to','book', 'ticket', 'to', 'Australia' ],
        ['I', 'want', 'to' ,'read', 'a' ,'book', 'of' ,'Shakespeare']
       ]

list(bigrams(text[0]))

list(ngrams(text[1], n=3))

'''
    (Now let's try an implementation of the n-gram in text generation,
    for this let's import the Trump tweet database from Kaggle and put it in a data frame.)

    Using panda library (Kaggle) to read stream from Donald Trump csv
    
    tumpe_corups = Tokenizing the tweet column

    apply n-grams to the corups by training the data and padded sentences using padded_everygram_pipeline

    using MLE - train size of 3 and find the maxiumum likelihood model from the trump-tweets

'''



df = pd.read_csv('../input/trump-tweets/realdonaldtrump.csv')
df.head()

trump_corpus = list(df['content'].apply(word_tokenize))

n = 3
train_data, padded_sents = padded_everygram_pipeline(n, trump_corpus)

trump_model = MLE(n)
set n=3
trump_model.fit(train_data, padded_sents)


'''

 Using Tree bank tonkenizer (treebank) from nltk,

 detokenizing after the sentences from the model has been generated.

 Keep track of the token pads for both <s> and </s> 


'''


detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


'''
    Randomizing seed randomly and gives the result from the Trump's tweets that has been generated.

    using N-grams can be beneficial to keep of track of accurate for every sequence of words, but making sure we should keep track of stop words.

    Might hard to predict is to keep of track of grammar when reading big text from either texfile, csv. and etc. 

'''

generate_sent(trump_model, num_words=20, random_seed=42)

generate_sent(trump_model, num_words=10, random_seed=0)









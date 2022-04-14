


import math
import pandas as pd
import numpy as np
import csv



#documents
doc1 = "Sir Ken Robinson makes an entertaining and profoundly moving case for creating an education system that nurtures (rather than undermines) creativity."
doc2 = "With the same humor and humanity he exuded in. An Inconvenient Truth, Al Gore spells out 15 ways that individuals can address climate change immediately, from buying a hybrid to inventing a new, hotter brand name for global warming."
doc3 = "New York Times columnist David Pogue takes aim at technology as worst interface-design offenders, and provides encouraging examples of products that get it right. To funny things up, he bursts into song."
#query string
query = "life learning"

#term -frequenvy :word occurences in a document
def compute_tf(docs_list):
    for doc in docs_list:
        doc1_lst = doc.split(" ")
        wordDict_1= dict.fromkeys(set(doc1_lst), 0)

        for token in doc1_lst:
            wordDict_1[token] +=  1
        df = pd.DataFrame([wordDict_1])
        idx = 0
        new_col = ["Term Frequency"]    
        df.insert(loc=idx, column='Document', value=new_col)
        print(df)
        

print("In [3]: ", compute_tf([doc1, doc2, doc3]))

#Normalized Term Frequency
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))

def compute_normalizedtf(documents):
    tf_doc = []
    for txt in documents:
        sentence = txt.split()
        norm_tf= dict.fromkeys(set(sentence), 0)
        for word in sentence:
            norm_tf[word] = termFrequency(word, txt)
        tf_doc.append(norm_tf)
        df = pd.DataFrame([norm_tf])
        idx = 0
        new_col = ["Normalized TF"]    
        df.insert(loc=idx, column='Document', value=new_col)
        print(df)
    return tf_doc

tf_doc = compute_normalizedtf([doc1, doc2, doc3])

print("In [4]: ", tf_doc)



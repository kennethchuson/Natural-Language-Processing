import numpy as np
import pandas as pd 


df = pd.read_csv('train.csv')

df.dropna(axis=0)
df.set_index('id', inplace = True)

df.head()

print("Out[1]: ", df.head()) 
###########################################################################################################


import re
import nltk 
from nltk.corpus import stopwords

#Using these libraries to determine number of verbs, adjectives, and nouns 
from nltk import pos_tag, word_tokenize

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


#nltk.download('stopwords')
#nltk.download('punkt')


stopWords = set(stopwords.words('english'))

convert_text_str_type = str(df['text'])
tokenize_each_text = word_tokenize(convert_text_str_type) 
pos_tag_text = pos_tag(tokenize_each_text)


store_verbs = [] 
store_nouns = [] 
store_adjectives = [] 


for (word, tag) in pos_tag_text: 
    if tag == 'RB': 
        store_verbs.append(word)  
    if tag == 'NN':
        store_nouns.append(word) 
    if tag == 'JJ': 
        store_adjectives.append(word) 



#creating a function to encapsulate preprocessing, to mkae it easy to replicate on  submission data
def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))

    
    #numerical feature engineering
    #total length of sentence
    df['length'] = df['processed'].apply(lambda x: len(x))

    
    #get number of words
    df['words'] = df['processed'].apply(lambda x: len(x.split(' ')))


    df['words_not_stopword'] = df['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
    #get the average word length
    df['avg_word_length'] = df['processed'].apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)
    #get the average word length
    df['commas'] = df['text'].apply(lambda x: x.count(','))

    
    #I addded number of adjectives, nouns, and verbs
    df['num_adjective_word'] = df['processed'].apply(lambda x: len(store_adjectives)) 

    df['num_noun_word'] = df['processed'].apply(lambda x: len(store_nouns)) 

    df['num_verb_word'] = df['processed'].apply(lambda x: len(store_verbs)) 

    return(df)

df = processing(df)

df.head()

print("Out[2]: ", df.head())

###########################################################################################################

from sklearn.model_selection import train_test_split

features= [c for c in df.columns.values if c  not in ['id','text','author']]
numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)


print("Out[3]: ", X_train.head())

###########################################################################################################



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

###########################################################################################################

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

text = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('tfidf', TfidfVectorizer( stop_words='english'))
            ])


print("Out[5]: ", text.fit_transform(X_train))

###########################################################################################################


from sklearn.preprocessing import StandardScaler

length =  Pipeline([
                ('selector', NumberSelector(key='length')),
                ('standard', StandardScaler())
            ])

length.fit_transform(X_train)


print("Out[6]: ", length.fit_transform(X_train))

###########################################################################################################


words =  Pipeline([
                ('selector', NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
words_not_stopword =  Pipeline([
                ('selector', NumberSelector(key='words_not_stopword')),
                ('standard', StandardScaler())
            ])
avg_word_length =  Pipeline([
                ('selector', NumberSelector(key='avg_word_length')),
                ('standard', StandardScaler())
            ])
commas =  Pipeline([
                ('selector', NumberSelector(key='commas')),
                ('standard', StandardScaler()),
            ])

###########################################################################################################


from sklearn.pipeline import FeatureUnion

feats = FeatureUnion([('text', text), 
                      ('length', length),
                      ('words', words),
                      ('words_not_stopword', words_not_stopword),
                      ('avg_word_length', avg_word_length),
                      ('commas', commas)])

feature_processing = Pipeline([('feats', feats)])


print("Out[8]: ", feature_processing.fit_transform(X_train))

print(feature_processing.fit_transform(X_train)) 



###########################################################################################################


from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('features',feats),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

print("Out[9]: ", np.mean(preds == y_test))

###########################################################################################################


print("Out[10]: ", pipeline.get_params().keys())

###########################################################################################################


from sklearn.model_selection import GridSearchCV



hyperparameters = { 'features__text__tfidf__max_df': [0.9, 0.95],
                    'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                   'classifier__max_depth': [50, 70],
                    'classifier__min_samples_leaf': [1,2]
                  }
#I changed it from 5 to 3 cross vaklidation for the experiment. 
clf = GridSearchCV(pipeline, hyperparameters, cv=3)
 
clf.fit(X_train, y_train)


print("Out[11]: ", clf.best_params_) 

###########################################################################################################


clf.refit


preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

print("Out[12]: ", np.mean(preds == y_test))

###########################################################################################################


submission = pd.read_csv('test.csv')

submission = processing(submission)
predictions = clf.predict_proba(submission)

preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

result = pd.concat([submission[['id']], preds], axis=1)
result.set_index('id', inplace = True)
print("Out[13]: ", result.head())


###########################################################################################################






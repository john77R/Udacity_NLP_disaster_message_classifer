


import sys
import re
import numpy as np
import pandas as pd
import nltk
import pickle
import sqlite3
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


# In[10]:





def load_data(database_filepath):
    
    ''' Load an sql lite database for analysis
    
    INPUT:
    database_filepath - the path of the .db file (database) is required. This is the out put of the process_data.py pipeline
   
    
    OUTPUT:
    X - dataframe=  attribute variable (in this case is 1 column containing the message to be classified from disaster_message.csv )
    Y - dataframe = target variables (36 classifications) 
    category_names - headers or classifications 
    '''

    
    
    connection = create_engine('sqlite:///{}'.format(database_filepath))
   
    df = pd.read_sql_table('DisasterResponse', connection) 
    
    df = df.replace(to_replace='None', value=np.nan)
    
    df=df[df["message"]!='#NAME?']
    
  
    X = pd.Series(df['message'])
    Y = df.drop(['id','message','original','genre'], axis=1)
    
    category_names = Y.columns
    
    return X, Y, category_names
        


# X, Y, category_names =load_data()

# In[3]:


def tokenize(text):
    
    ''' Tokenizes the input text,
    removes stop words & removes all non-letter characters.
    
    INPUT:
    df- text our X variable
      
    
    OUTPUT:
     tokens- created from
    1]text converted to only letters.
    2]words are tokenized- split into seperate objects.
    3]remove english language stop words.
    4]words are then converted to their root form via lemmatization(+lowercase)
   
    '''
    
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
   
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove english language stop words
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token to lemmatize
    Cln_tokens = []#list to contain text tokens
    for tkn in tokens:
        
        # lemmatize, lower case and stip out variation in white space.
        token = lemmatizer.lemmatize(tkn).lower().strip()
        Cln_tokens.append(token)

    return Cln_tokens 
    


# In[12]:


def build_model():
    
    '''  Builds a sklearn pipeline model
    
    OUTPUT:
    model_pipeline -a text processing pipe line for machine learning classification:
            - CountVectorizer as vectorizer
            - TfidfTransformer as transformer
            - MultiOutputClassifier, with "RandomForestClassifier", as classifier
            - GridSearchCV in order to automate hyper parameter tuning and to find best set of hyperparmeters within the specified search space
    
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))   
    ])
    
    parameters = {
    
       'vect__ngram_range': [(1, 1)],
    
       'vect__max_df': [0.5],
       'vect__max_features': [None],

        'clf__estimator__n_neighbors': [2]
                } #this is our search space
    
    
    cv = GridSearchCV(pipeline, parameters,verbose=3)
    print(cv)
    return cv


# In[13]:


def train(X_train, y_train, model):
    
    ''' Fits the model with the training data
    
    INPUT:
    X_train & y_train from our test train split function
    model - machine learning model to predict classifications from training data.
   
    OUTPUT:
    Model-trained model. 
   
    '''

    # fit model to training data
    model.fit(X_train, y_train)
    
    return model


# In[14]:


def evaluate_model(model, X_test, y_test, category_names):    
    
    ''' Evaluates the trained model,using the test set. ClassificationReport provides the metrics for each classification
    
    INPUT:
    model - machine learning model to predict classifications fit on the training data.

    X_test -X test data 
    y_test -y test data
 
   
    '''    
    
    ytest_predict = model.predict(X_test)

    # convert the model predictions in numpyarrays into a dataframe for futher processing.

    y_headers = y_test.columns

    df_ytest_predict = pd.DataFrame(ytest_predict, columns = y_headers)

    #itterate through the columns of the y-predict data frame and compare to the Ytest set using the classification report
    for col in y_test:
        print("Test Score results for Category..........",col)
        test_score = classification_report(y_test[col],df_ytest_predict[col])
        #print("\nBest Parameters:", cv.best_params_)
        print(test_score)


# In[15]:


def save_model(model, model_filepath):
    
    ''' Saves the best trained model as a pickle file
    
    INPUT:
    model  
    model_filepath - location where model will be saved
   
    '''       
    
    pkl_filename = '{}'.format(model_filepath)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


# In[16]:


def main():
    
    ''' Main Function brings togeather various subfunctions to build a Machine learning Pipeline for text data.
    Model optimisation is preformed with CV grid search. The best model is saved as a pickle file.
    
    INPUT:
    database_filepath - path of the .db file (database) created and stored by the process_data.py script 
    model_filepath - path where model will be saved
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')




if __name__ == '__main__':
    main()


# In[ ]:





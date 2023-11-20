import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle

nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

#asd

def load_data(database_filepath):
    '''
    Loads data from SQL into a pandas dataframe, separates the variables into different dataframes
    input:
    database_filepath - the location of the database file

    output
    X - explanatory variable
    y - categories for classification
    y.columns - category names
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM database", engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X,y,y.columns

def tokenize(text):
    '''
    Takes a text, removes the non-alphanumeric characters, removes stopwords, runs lemmatization and tokenization
    input:
    text - string to tokenize
    output:
    tokens - created tokens from the string
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    stop_words = stopwords.words("english")
    lemmatizer= WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    Creates a machine learning classification pipeline
    Runs gridsearchcv for parameter tuning
    
    output:
    pipeline - machine learning pipeline
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))#,('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3)))
    ])
    

    parameters = { 
        'clf__estimator__n_estimators': [25, 50,75],
        'clf__estimator__learning_rate' : [0.5,1,1.5,0.2]
    }
    cv = GridSearchCV(pipeline,param_grid=parameters, n_jobs=-1)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model performance using classification report
    input:
    model - ML model pipeline
    X_test - test messages
    Y_test - test categories
    category_names - name of the groups for printing the report

    output:
    classification report showing precision, recall, f1-score
    '''
    Y_test_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_test_pred,target_names=category_names))

def save_model(model, model_filepath):
    '''
    saves the ml model to a pickle file
    input:
    model - ml pipeline
    model_filepath - pickle file save location
    '''
    with open (model_filepath,'wb') as file:
        pickle.dump(model,file)


def main():
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
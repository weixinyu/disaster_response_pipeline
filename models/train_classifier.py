import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sqlite3
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM MessagesWithCategories', con = conn)
    
    X = df['message'].values
    y = df.iloc[:, 4:].values
    labels = df.columns[4:].tolist()
    return X, y, labels


def tokenize(text):
    '''
        normalize, tokenize and lemmatize
        Input:
            text: original message text
        Output:
            tokens: Cleaned words list
    '''
    # tokenize
    tokens = word_tokenize(text)
    
    # normailize and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]

    return tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier())
    ])

    # specify parameters for grid search
    parameters = {
        'clf__n_estimators': [50],
        'clf__min_samples_split': [2]

    } 

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    classification_result = classification_report(Y_test, Y_pred, target_names =category_names)
    print(classification_result)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
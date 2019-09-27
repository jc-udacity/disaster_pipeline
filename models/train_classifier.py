import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """ This function is loading data from a database file

    Parameters
    ----------
    database_filepath : string
        filepath of the data file `database_filepath`.

    Returns
    -------
    dataframe X: messages
    dataframe Y: features
    list category_names: list of category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessageTable', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message','original','genre' ])
    category_names = list(Y.columns.values)

    return X, np.array(Y), category_names


def tokenize(text):
    """ This function is tokenizing the text, removing stopwords
        and finally performing a lemmatize step on the tokens

    Parameters
    ----------
    text : string
        Text to be tokenized

    Returns
    -------
    clean_tokens: list
        List of cleaned tokens

    """
    # tokenize
    tokens = word_tokenize(text)

    # Remove Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ This function is building a model with a pipeline and
        some parameters for GridSearchCV

    Returns
    -------
    model : GridSearchCV object

    """
    #building a pipeline with 3 steps: vectorize, transform, classify
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators': [30,50,100],
        'clf__estimator__max_depth': [3, 5],
        'clf__estimator__min_samples_split': [3,5],
        'clf__estimator__criterion' : ['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function fit the model append
       evaluate it with classification report outputs.

    Parameters
    ----------
    model : GridSearchCV object
        This is the model object that is evaluated
    X_test : This is the messages dataframe
    Y_test : This is the features List
    category_names : This is the List of category names

    Returns
    -------
    This function doesn't return anything.
    It only prints the classification report in stdout

    """
    y_pred = model.fit(X_test)
    for category in range(len(category_names)):
        print('category: {}'.format(category_names[category]))
        print(classification_report(Y_test[:, category], y_pred[:, category]))


def save_model(model, model_filepath):
    """ This function save the model in a file

    Parameters
    ----------
    model : GridSearchCV model
    model_filepath : full path and filename where to save the model

    Returns
    -------
    This function doesn't return anything. It only save the model in a file

    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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

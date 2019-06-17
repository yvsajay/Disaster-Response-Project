import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Input - A database_filepath which has a table named 'msg_categories'.
    Output - An array X with messages, a dataframe y with categories, and a list of category names.
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('msg_categories',engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names=list(df.iloc[:,4:].columns)
    return X, y, category_names

def tokenize(text):
    '''
    Input - text 
    Output - Clean tokens which are words in the text that are first normalized in case,
    and then the stop words are removed after lemmatizing.
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    token = word_tokenize(text)
    
    clean_token = []
    for tok in token:  
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_token.append(clean_tok)
    return clean_token


def build_model():
    '''
    Builds a MultiOutputClassifier using RandomForesrClassifier. Uses pipelines and GridSearchCV to optimize the parameters.
    Uses CountVectorizer, TfidfTransformer in the pipeline along with the classifiers mentioned above.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, verbose=1), n_jobs= -1))
    ])
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 4]
            }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1) #Define GridSearchCV
    return model
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''Returns the f1 score, precision and recall of a trained model by predicting on its test dataset. 
    Input - Trained model, It's test dataset - both input variables and target variables, and the list of target names
    Output - F1 score, precision and Recall for each of the target variables
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    return


def save_model(model, model_filepath):
    '''Save the trained model in a provided path location
    Input - A trained model and a filepath location where the file has to be saved.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return


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

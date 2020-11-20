# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
import joblib

# Function to load data
def load_data(database_filepath):
    """
    Load the data
    
    Arguments:
      database_filepath(string): the file path of input database
      
    Return:
      X (pandas dataframe): Features dataframe
	  y (pandas dataframe): Targets dataframe
      category_names (list): List of target names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, y, category_names

# Create class to create starting verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
# Create class to normalize the text
class CaseNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.Series(X).apply(lambda x: x.lower()).values

# Function to create clean tokens through text processing
def tokenize(text):
    """
    Returns the word tokens after reducing words to their root form
    
    Args:
        text(string): input message text
    Returns:
        clean_tokens (list): list of reduced words to their root form
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Function to develop the pipeline
def develop_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('lowercase', CaseNormalizer()),
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

# Function to develop the model
def build_model():
    """
    Returns the grid search model based on the parameter space defined in the function
    
    Args:
        None
    Returns:
        cv: Grid search model
    """
    parameters = {
                    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
                    'clf__estimator__n_estimators': [100, 200]
                 }
    cv = GridSearchCV(develop_pipeline(), param_grid=parameters)
    return cv

# Function to evaluate the model and store the results to a dataframe
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns multi-output classification results dataframe
    
    Args:
        model (dataframe): the scikit-learn fitted model
        X_text (dataframe): Feature test dataframe
        y_test (dataframe): Target test dataframe
        category_names (list): Category names
    Returns:
        results (dataframe): Model results containing precision, recall and f-score per output
    """
    y_pred = model.predict(X_test)
    count = 0
    results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f_score'])
    for category in category_names:
        precision, recall, f_score, support = score(Y_test[category], y_pred[:,count], average='weighted')
        results.at[count, 'category'] =category
        results.at[count, 'precision'] = precision
        results.at[count, 'recall'] = recall
        results.at[count, 'f_score'] = f_score
        count += 1
    return results

# Function to output the model for further use
def save_model(model, model_filepath):
    """
    Saves the developed model to given path 
    
    Args: 
        model (estimator): The fitted model
        model_filepath (str): Filepath to save the model
    Return:
        None
	"""
    joblib.dump(model, model_filepath)


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
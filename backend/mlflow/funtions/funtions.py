import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
import string
import re
import mlflow
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from datetime import datetime
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import chain
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
from sklearn import model_selection as ms
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

def fetch_data():

    print('Extraccion del dataset')
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    website_df = pd.read_csv('funtions/website_classification.csv')
    website_url_df = website_df[['website_url', 'Category']]
    website_df.drop(['Unnamed: 0', 'website_url'], axis=1, inplace=True)

    plt.figure(figsize=(12, 10))
    plt.title('Category vs Counts')
    website_category_df = pd.DataFrame(website_df.groupby('Category').size(), columns=['Count'])
    plt.barh(width=website_category_df['Count'], y=website_category_df.index)
    plt.show()

    website_df['tokenized_words'] = website_df['cleaned_website_text'].apply(lambda x: word_tokenize(x))
    website_df['tokenized_words'] = website_df['tokenized_words'].apply(
        lambda x: [re.sub(f'[{string.punctuation}]+', '', i) for i in x if i not in list(string.punctuation)])

    website_df['tokenized_words'] = website_df['tokenized_words'].apply(
        lambda x: [i for i in x if i not in stopwords.words('english')])
    wordnetlemmatizer = WordNetLemmatizer()
    website_df['tokenized_words'] = website_df['tokenized_words'].apply(
        lambda x: [wordnetlemmatizer.lemmatize(i) for i in x])
    website_df['tokenized_words'] = website_df['tokenized_words'].apply(lambda x: ' '.join(x))
    website_df.drop(['cleaned_website_text'], axis=1, inplace=True)
    website_df = website_df[['tokenized_words', 'Category']]
    website_df.columns = ['Website Cleaned Text', 'Category']

    print('enconding de las categorias')
    le = LabelEncoder()
    website_df['Category'] = le.fit_transform(website_df['Category'])

    X_train, X_test, y_train, y_test = train_test_split(website_df['Website Cleaned Text'], website_df['Category'],
                                                        test_size=0.3, random_state=0)

    print('extraccion de datos')
    X_train.to_csv('X_train.csv')

    tf_idf_vectorizer = TfidfVectorizer()
    X_train = tf_idf_vectorizer.fit_transform(X_train)
    X_train = pd.DataFrame(X_train.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())

    X_test = tf_idf_vectorizer.transform(X_test)
    X_test = pd.DataFrame(X_test.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())

    sampling_strategy = {14: 81, 4: 77, 3: 76, 13: 75, 8: 75, 7: 72, 15: 70, 1: 69, 2: 69, 12: 65, 9: 62, 10: 59, 5: 58,
                         11: 54, 0: 60, 6: 60}
    oversample = RandomOverSampler(sampling_strategy=sampling_strategy)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def create_experiment(experiment_name):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def generate_confusion_matrix_figure(model_name, model,X_train, X_test, y_train, y_test):

    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test

    try:
        check_is_fitted(model)
    except NotFittedError:
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Confusion Matrix for {model_name}")
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    return fig

def run_experiment(experiment_id, n_splits=5):
    count = 1
    if count == 1:
        X_train, X_test, y_train, y_test = fetch_data()
    count += 1

    models = {
        'MNB': MultinomialNB(alpha=0.5,fit_prior=False),
        'SVM': SVC(C=1000,kernel='rbf',gamma=0.001),
        'RFC': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        'GNB': GaussianNB(),
    }

    # Run each model in a separate run
    for model_name, model in models.items():
        print(f"Running {model_name}...")

        # create a unique run id
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{model_name}-{run_id}"

        # start a new run in MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as run:
            kfold = model_selection.KFold(n_splits=n_splits, random_state=7, shuffle=True)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
            print(f"Accuracy: {cv_results.mean():.3f} ({cv_results.std():.3f})")

            # Log the model accuracy to MLflow
            mlflow.log_metric("accuracy", cv_results.mean())
            mlflow.log_metric("std", cv_results.std())
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_splits", n_splits)

            for fold_idx, kflod_result in enumerate(cv_results):
                mlflow.log_metric(key="crossval", value=kflod_result, step=fold_idx)

            # # fit model on the training set and log the model to MLflow

            model.fit(X_train, y_train)
            signature = infer_signature(X_train, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature
            )
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
            # log artifacts
            fig = generate_confusion_matrix_figure(model_name, model, X_train, X_test, y_train, y_test)
            mlflow.log_figure(fig, f"{model_name}-confusion-matrix.png")

def get_best_run(experiment_id, metric):
    """
    Get the best run for the experiment
    :param experiment_id:  id of the experiment
    :param metric:  metric to use for comparison
    :return:
    """
    client = MlflowClient()

    # Get all the runs for the experiment
    runs = client.search_runs(experiment_id)

    # Find the run with the highest accuracy metric
    best_run = None
    best_metric_value = 0
    for run in runs:
        metric_value = run.data.metrics[metric]
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run = run
    # Return the best run
    return best_run

def get_best_model(experiment_id, metric):
    """
    Get the best model for the experiment
    :param experiment_id:
    :param metric:
    :return:
    """
    # Get the best run
    best_run = get_best_run(experiment_id, metric)
    # Get the model artifact URI
    model_uri = f"runs:/{best_run.info.run_id}/{best_run.data.params['model_name']}"
    # Load the model as a PyFuncModel
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def register_best_model(experiment_id, metric, registered_model_name):
    """
    Register the best model in the experiment as a new model in the MLflow Model Registry
    :param experiment_id:
    :param metric:
    :param registered_model_name:
    :return:
    """
    # Get the best run
    best_run = get_best_run(experiment_id, metric)
    # Get the model artifact URI
    model_uri = f"runs:/{best_run.info.run_id}/{best_run.data.params['model_name']}"
    # registered_model = find_model_by_name(registered_model_name)
    # if registered_model is None:
    registered_model = mlflow.register_model(model_uri, registered_model_name)
    return registered_model

def promote_model_to_stage(registered_model_name, stage, version=None):
    """
    Promote the latest version of a model to the given stage
    :param registered_model_name:
    :param stage:
    :return:
    """
    client = MlflowClient()
    model = client.get_registered_model(registered_model_name)
    if version is not None:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage,
        )
        return
    latest_versions = [mv.version for mv in model.latest_versions]
    client.transition_model_version_stage(
        name=registered_model_name,
        version=max(latest_versions),
        stage=stage,
    )

def call_model_at_stage(registered_model_name, stage, data):
    """
    Call the production model to get predictions
    :param registered_model_name:
    :param stage:
    :param data:
    :return:
    """
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{registered_model_name}/{stage}"
    )
    # Evaluate the model
    predictions = model.predict(data)
    return predictions

def funtions():
    return None
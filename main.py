import argparse
import os
import pickle
import nltk
import pandas as pd
from nltk.corpus import twitter_samples
from utils import process_tweets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


parser = argparse.ArgumentParser(description='TwitterSentimentAnalysis')
parser.add_argument('--model', type=str, default='LR',
                    help='[LR, BNB, LSVC]')
parser.add_argument('--train',action="store_true",
                    help='wheter to train the model or not')
args = parser.parse_args()


# downloads sample twitter dataset.
nltk.download('twitter_samples')
nltk.download('stopwords')


def load_data():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    processed_positive_tweets = process_tweets(all_positive_tweets)
    processed_negative_tweets = process_tweets(all_negative_tweets)
    all_processed_tweets = processed_positive_tweets + processed_negative_tweets
    sentiment = np.append(np.ones((len(processed_positive_tweets), 1)), np.zeros((len(processed_negative_tweets), 1)))

    return all_processed_tweets, sentiment


def get_model(model_name='LR'):

    if model_name == 'LR':
        return LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    elif model_name == 'BNB':
        return  BernoulliNB(alpha = 2)
    elif model_name == 'LSVC':
        return  LinearSVC()


def train_model(model_name):

    processed_tweets, sentiment = load_data()

    x_train, x_test, y_train, y_test = train_test_split(processed_tweets, sentiment,
                                                        test_size=0.05, random_state=0)
    # TF-IDF Vectoriser converts a collection of raw documents to a matrix of TF-IDF features.
    # The Vectoriser is usually trained on only the X_train dataset.
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(x_train)
    # print(f'Vectoriser fitted.')
    # print('No. of feature_words: ', len(vectoriser.get_feature_names()))

    x_train = vectoriser.transform(x_train)
    x_test = vectoriser.transform(x_test)

    model = get_model(model_name)
    model.fit(x_train, y_train)
    os.makedirs('./model', exist_ok=True)
    file = open('./model/vectoriser-ngram-(1,2).pickle', 'wb')
    pickle.dump(vectoriser, file)
    file.close()

    file = open('./model/sentiment-'+model_name+'.pickle', 'wb')
    pickle.dump(model, file)
    file.close()

    evaluate_model(model, x_test, y_test)
    return vectoriser, model


def load_model(model_name):

    # Load the vectoriser.
    file = open('./model/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./model/sentiment-'+model_name+'.pickle', 'rb')
    model = pickle.load(file)
    file.close()

    return vectoriser, model


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(process_tweets(text))
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


def evaluate_model(model, x, y):
    # Predict values for Test dataset
    y_pred = model.predict(x)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y, y_pred))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)

    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    # plt.show()


if __name__ == "__main__":
    # Loading the models.
    model_name = args.model

    if args.train:
        vectoriser, LRmodel = train_model(model_name)
    else:
        vectoriser, LRmodel = load_model(model_name)

    # Text to classify should be in a list.
    text = ["Happy coding!",
            "We hope you enjoy! Happy Friday!",
            "I don't feel so good."]

    df = predict(vectoriser, LRmodel, text)
    print(df.head())

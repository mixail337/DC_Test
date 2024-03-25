import csv

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import json
from deeppavlov import configs, train_model, build_model


def load_marked_data(annot_data):
    with open(annot_data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('data.txt', 'w', encoding='utf-8') as f:
        for item in data:
            text = item['text']
            labels = item['label']
            words = nltk.word_tokenize(text)
            tags = {word: 'O' for word in words}
            for label in labels:
                start = label['start']
                end = label['end']
                if " " in text[start:end]:
                    print(text[start:end])
                    st, nd = text[start:end].split()
                    tags[st] = 'B-LOC'
                    tags[nd] = 'I-LOC'
                else:
                    tags[text[start:end]] = 'B-LOC'

            for word, tag in tags.items():
                f.write(f'{word} {tag}\n')
            f.write('\n')


def split_data():
    with open('data.txt', 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n\n')
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(item + '\n\n')
    with open('data/valid.txt', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(item + '\n\n')


def training_ner(config):
    with open(config, 'r') as f:
        data = json.load(f)
    data['dataset_reader']['data_path'] = 'data'
    with open(config, 'w') as f:
        json.dump(data, f, indent=4)
    ner_config = config
    ner_model = train_model(ner_config)


def get_predictions_ner(config, messages):
    ner_model = build_model(config, download=False)
    predicted = []
    for message in messages:
        prediction = ner_model([message])
        tokens = prediction[0][0]
        labels = prediction[1][0]

        cities = []
        current_city = ''

        for token, label in zip(tokens, labels):
            if label == 'B-LOC':
                if current_city:
                    cities.append(current_city)
                current_city = token
            elif label == 'I-LOC' and current_city:
                current_city += ' ' + token

        if current_city:
            cities.append(current_city)

        city = ''.join(cities)
        while len(city) > 0 and not city[-1].isalpha():
            city = city[:-1]
        predicted.append(city)
    return predicted


def save_to_txt(predicted, answers):
    with open('datas.txt', 'w') as f:
        for pred, ans in zip(predicted, answers):
            f.write(pred + ' ' + ans + '\n')


def train_svc(predicted, answers):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(predicted, answers, test_size=0.2,
                                                                              random_state=42)
    vectorizer = CountVectorizer()
    inputs_train = vectorizer.fit_transform(inputs_train)
    inputs_test = vectorizer.transform(inputs_test)
    classifier = SVC()
    classifier.fit(inputs_train, targets_train)
    joblib.dump(classifier, 'model1.pkl')
    joblib.dump(vectorizer, 'vectorizer1.pkl')


def get_cities_svc(messages):
    classifier = joblib.load('model1.pkl')
    vectorizer = joblib.load('vectorizer1.pkl')
    cities = []
    for message in messages:
        cities.append(classifier.predict(vectorizer.transform([message]))[0])
    return cities


def custom_input(config, message):
    ner_model = build_model(config, download=False)
    prediction = ner_model([message])
    tokens = prediction[0][0]
    labels = prediction[1][0]
    cities = []
    current_city = ''
    for token, label in zip(tokens, labels):
        if label == 'B-LOC':
            if current_city:
                cities.append(current_city)
            current_city = token
        elif label == 'I-LOC' and current_city:
            current_city += ' ' + token

    if current_city:
        cities.append(current_city)

    city = ''.join(cities)
    while len(city) > 0 and not city[-1].isalpha():
        city = city[:-1]

    classifier = joblib.load('model1.pkl')
    vectorizer = joblib.load('vectorizer1.pkl')
    result = classifier.predict(vectorizer.transform([city]))[0]
    print(result)


def save_results(results, messages):
    with open('data.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'message', 'city'])
        for i in range(len(results)):
            writer.writerow([i, messages[i], results[i]])


if __name__ == '__main__':
    # data = pd.read_csv('data123.csv', delimiter=';')
    # messages = data['message'].tolist()
    # answers = data['Answer'].tolist()
    #
    ner_config = 'ner_rus_bert_new.json'
    # # training_ner(ner_config)
    # predictions = get_predictions_ner(ner_config, messages)
    # # save_to_txt(predictions, answers)
    # # train_svc(predictions,answers)
    # results = get_cities_svc(messages)
    # save_results(results, messages)
    message = 'Привет, хочу заказать доставку цветов в новосиб, какая цена?'
    custom_input(ner_config, message)



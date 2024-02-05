import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()

def load_model_file(file_path):
    try:
        return load_model(file_path)
    except FileNotFoundError:
        print(f"Error: Model file '{file_path}' not found.")
        exit()

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Pickle file '{file_path}' not found.")
        exit()

def clean_up_sentence(sentence, lemmatizer):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence, lemmatizer)
    return np.array([1 if word in sentence_words else 0 for word in words])

def predict_class(sentence, model, words):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    confidence = float(intents_list[0]['probability'])
    list_of_intents = intents_json['intents']

    filtered_intents = [i for i in list_of_intents if i['tag'] == tag]

    if confidence < 0.5 or not filtered_intents:
        new_intent = {"tag": tag, "patterns": [], "responses": [f"I do not understand. Can you please rephrase or ask a different question about {tag}?"]}
        list_of_intents.append(new_intent)

        with open('intents.json', 'w') as json_file:
            json.dump(intents_json, json_file, indent=2)

        return f"I have added a new category for {tag}. I do not understand questions about {tag} yet. Can you provide more information?"

    return random.choice(filtered_intents[0]['responses'])

def main():
    global lemmatizer, classes

    lemmatizer = WordNetLemmatizer()
    intents = load_data('intents.json')
    words = load_pickle('words.pkl')
    classes = load_pickle('classes.pkl')
    model = load_model_file('chatbot_model.h5')

    print("Bot is running!")

    while True:
        message = input("You: ")
        ints = predict_class(message, model, words)
        response = get_response(ints, intents)

        if response.startswith("I have added a new category for"):
            print("Bot:", response)
        else:
            print("Bot:", response)

if __name__ == '__main__':
    main()

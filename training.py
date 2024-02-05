import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract words and classes from intents
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend([lemmatizer.lemmatize(word.lower()) for word in word_list])
        documents.append((word_list, intent['tag']))

# Remove duplicates and sort words and classes
words = sorted(set(words))
classes = sorted(set(intent['tag'] for intent in intents['intents']))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [1 if lemmatizer.lemmatize(word.lower()) in document[0] else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert training data to NumPy array
random.shuffle(training)
training = np.array(training)

# Split data into features and labels
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print('Chatbot training completed.')

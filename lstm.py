import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Need to read as windows string else it would throw an error.
df = pd.read_csv("gmb.csv", sep=",",encoding="cp1252")

words = df.Word.to_list()
part_of_speech = df.POS.to_list()
named_entity = df.Tag.to_list()

# the very important tokanizer
tokenizer = Tokenizer(char_level=True)
# fitting the tokenizer on the words
tokenizer.fit_on_texts(words)
# length of the vocabulary in the training set
vocab_size = len(tokenizer.word_index)+1

words = tokenizer.text_to_sequences(words)

MAX_LENGTH = max(len(word) for word in words)
words = pad_sequences(words, maxlen=MAX_LENGTH, padding="post")

pos_classes = len(part_of_speech.unique)
ne_classes = len(named_entity.unique)

# splitting the dataset into 2 parts
# 10% testing
# 90% training
pos_x_train, pos_x_test, pos_y_train, pos_y_test = train_test_split(words, part_of_speech, test_size=0.10, random_state=69)
ne_x_train, ne_x_test, ne_y_train, ne_y_test = train_test_split(words, named_entity, test_size=0.10, random_state=69)

#
# I have no Idea how to test with two labels as prediction...
#
lstm = Sequential()
lstm.add(Embedding())
lstm.add(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))
lstm.add(Dense(1, activation='softmax'))

print(classification_report())
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# attempted to realise a prediction of a sequence of target labels (POS_Tag labels) for a every input sequence (the
# sentences) but it doesn't work well

# reading in only works if I set encoding to Latin-1
df = pd.read_csv("gmb.csv", sep=",", encoding="latin-1")

# fill all rows with their sentence number for grouping (ffill puts last valid entry until next valid)
df['Sentence #'] = df['Sentence #'].fillna(method='ffill')
df = df.dropna(subset=['Word', 'POS', 'Tag'])
df['POS_Tag'] = df['POS'] + "_" + df['Tag']

# will create a list containing all sentences as a list of their words (pd apply() = use a function on the df)
sentences = df.groupby('Sentence #')['Word'].apply(list).tolist()

# same for the labels of the individual words in each sentence
labels = df.groupby('Sentence #')['POS_Tag'].apply(list).tolist()

# tokenize sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
X_sequences = tokenizer.texts_to_sequences(sentences)

# flatten labels into a singular list and use LabelEncoder to fit and transform the POS_Tag labels into numerical values
flattened_labels = [label for sentence_labels in labels for label in sentence_labels]
label_encoder = LabelEncoder()
label_encoder.fit(flattened_labels)
Y_sequences = [label_encoder.transform(label_seq) for label_seq in labels]

# padding
MAX_LENGTH = max(len(sentence_seq) for sentence_seq in X_sequences)
X_sequences = pad_sequences(X_sequences, maxlen=MAX_LENGTH, padding='post')
Y_sequences = pad_sequences(Y_sequences, maxlen=MAX_LENGTH, padding='post')

# one-hot encode labels
num_classes = len(label_encoder.classes_)
Y_sequences = [to_categorical(y, num_classes=num_classes) for y in Y_sequences]
X_sequences = np.array(X_sequences)
Y_sequences = np.array(Y_sequences)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sequences, Y_sequences, test_size=0.10,
                                                    random_state=1337)
# define LSTM
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 64, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                             kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4),
                             activity_regularizer=L1(1e-5), recurrent_regularizer=L1L2(l1=1e-5, l2=1e-4))))
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy")
model.summary()

# # RNN:
# model = Sequential()
# model.add(Input(shape=(MAX_LENGTH,)))
# model.add(Embedding(vocab_size, 150, input_length=MAX_LENGTH))
# model.add(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3,
#                     return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4),
#                     activity_regularizer=L1(1e-5), recurrent_regularizer=L1L2(l1=1e-5, l2=1e-4)))
# model.add(Dense(num_classes, activation="softmax"))
# model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.001))
# model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = np.argmax(y_test, axis=-1)
# need to be a flat representation for evaluation again
y_pred_flat = [label for seq in y_pred for label in seq]
y_test_flat = [label for seq in y_test for label in seq]
# padding removal
non_pad_indices = np.where(np.array(y_test_flat) != 0)
y_pred_final = np.array(y_pred_flat)[non_pad_indices]
y_test_final = np.array(y_test_flat)[non_pad_indices]

# Classification report
print(classification_report(y_test_final, y_pred_final))



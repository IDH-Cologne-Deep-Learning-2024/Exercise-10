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

df = pd.read_csv("gmb.csv", sep=",", encoding="latin1")
df = df.dropna(subset=[col for col in ["Word", "POS", "Tag"] if col in df.columns])
df["Word"] = df["Word"].astype(str)
df['POS'] = pd.factorize(df['POS'])[0]
df['Tag'] = pd.factorize(df['Tag'])[0]

sentences = df.groupby('Sentence #')['Word'].apply(list).tolist()
pos_labels = df.groupby('Sentence #')['POS'].apply(list).tolist()
ne_labels = df.groupby('Sentence #')['Tag'].apply(list).tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([item for sublist in sentences for item in sublist])
token_sequences = tokenizer.texts_to_sequences(sentences)
vocab_size = len(tokenizer.word_index)

MAX_LENGTH = max(len(seq) for seq in token_sequences)
token_sequences = pad_sequences(token_sequences, maxlen=MAX_LENGTH, padding='post')
pos_labels = pad_sequences(pos_labels, maxlen=MAX_LENGTH, padding='post', value=-1)
ne_labels = pad_sequences(ne_labels, maxlen=MAX_LENGTH, padding='post', value=-1)
pos_labels[pos_labels == -1] = 0
ne_labels[ne_labels == -1] = 0

X_train, X_test, y_train_pos, y_test_pos, y_train_ne, y_test_ne = train_test_split(
    token_sequences, pos_labels, ne_labels, test_size=0.1, random_state=42
)

def build_model(model_type, num_classes, input_length, vocab_size, embedding_dim=50):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=input_length))
    if model_type == 'LSTM':
        model.add(LSTM(128, return_sequences=True))
    elif model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
    elif model_type == 'RNN':
        model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

results = {}
for model_type in ['LSTM', 'BiLSTM', 'RNN']:
    print(f"Training {model_type} for POS tagging...")
    num_pos_classes = np.max(y_train_pos) + 1  
    model_pos = build_model(model_type, num_pos_classes, MAX_LENGTH, vocab_size)
    model_pos.fit(X_train, y_train_pos, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
    y_pred_pos = model_pos.predict(X_test).argmax(axis=-1)
    print(f"{model_type} POS classification report:\n")
    print(classification_report(y_test_pos.flatten(), y_pred_pos.flatten()))

    print(f"Training {model_type} for NE recognition...")
    num_ne_classes = np.max(y_train_ne) + 1  
    model_ne = build_model(model_type, num_ne_classes, MAX_LENGTH, vocab_size)
    model_ne.fit(X_train, y_train_ne, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
    y_pred_ne = model_ne.predict(X_test).argmax(axis=-1)
    print(f"{model_type} NE classification report:\n")
    print(classification_report(y_test_ne.flatten(), y_pred_ne.flatten()))

    results[model_type] = {
        'POS': classification_report(y_test_pos.flatten(), y_pred_pos.flatten(), output_dict=True),
        'NE': classification_report(y_test_ne.flatten(), y_pred_ne.flatten(), output_dict=True)
    }

for model_type, metrics in results.items():
    print(f"{model_type} F1 scores: POS - {metrics['POS']['macro avg']['f1-score']}, NE - {metrics['NE']['macro avg']['f1-score']}")



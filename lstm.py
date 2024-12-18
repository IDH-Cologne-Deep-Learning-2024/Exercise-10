import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def validate_csv(file_path):
    try:
        with open(file_path, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)
            header = next(reader)
            column_count = len(header)
            for row in reader:
                if len(row) != column_count:
                    print("Inconsistent row length detected")
                    return False
        return True
    except csv.Error as e:
        print(f"CSV Error: {e}")
        return False

def load_data(file_path):
    tokens, pos_tags, ne_tags = [], [], []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            tokens.append(row[0])
            pos_tags.append(row[1])
            ne_tags.append(row[2])
    return tokens, pos_tags, ne_tags

def preprocess_data(tokens, pos_tags, ne_tags):
    token_to_index = {token: idx for idx, token in enumerate(set(tokens))}
    pos_to_index = {tag: idx for idx, tag in enumerate(set(pos_tags))}
    ne_to_index = {tag: idx for idx, tag in enumerate(set(ne_tags))}
    
    X = [token_to_index[token] for token in tokens]
    y_pos = [pos_to_index[tag] for tag in pos_tags]
    y_ne = [ne_to_index[tag] for tag in ne_tags]
    
    X = pad_sequences([X], padding='post')
    y_pos = to_categorical(y_pos)
    y_ne = to_categorical(y_ne)
    
    return X, y_pos, y_ne, len(token_to_index), len(pos_to_index), len(ne_to_index)

def create_model(input_dim, output_dim, lstm_units=64, bidirectional=False):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=50, input_length=None))
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    else:
        model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X, y, model, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true.flatten(), y_pred.flatten(), average='weighted')
    recall = recall_score(y_true.flatten(), y_pred.flatten(), average='weighted')
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
    
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    file_path = "gmb.csv"
    
    if validate_csv(file_path):
        tokens, pos_tags, ne_tags = load_data(file_path)
        X, y_pos, y_ne, vocab_size, pos_size, ne_size = preprocess_data(tokens, pos_tags, ne_tags)
        
        print("Training LSTM for POS tagging")
        lstm_pos = create_model(vocab_size, pos_size)
        pos_metrics = train_and_evaluate(X, y_pos, lstm_pos)
        
        print("Training BiLSTM for NE tagging")
        bilstm_ne = create_model(vocab_size, ne_size, bidirectional=True)
        ne_metrics = train_and_evaluate(X, y_ne, bilstm_ne)
        
        print("POS Tagging Results (LSTM):")
        print(f"Accuracy: {pos_metrics[0]:.4f}")
        print(f"Precision: {pos_metrics[1]:.4f}")
        print(f"Recall: {pos_metrics[2]:.4f}")
        print(f"F1-score: {pos_metrics[3]:.4f}")
        
        print("\nNamed Entity Recognition Results (BiLSTM):")
        print(f"Accuracy: {ne_metrics[0]:.4f}")
        print(f"Precision: {ne_metrics[1]:.4f}")
        print(f"Recall: {ne_metrics[2]:.4f}")
        print(f"F1-score: {ne_metrics[3]:.4f}")
    else:
        print("CSV file is invalid. Please check the file and try again.")

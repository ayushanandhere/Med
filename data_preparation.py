import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv('train_data_chatbot.csv')


data = {
    "Question": df['short_question'].tolist(),  
    "Answer": df['short_answer'].tolist()       
}

df_prepared = pd.DataFrame(data)
df_prepared.to_csv('medical_chatbot_data.csv', index=False)
print("Data saved to medical_chatbot_data.csv")

# Tokenize the questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Question'])
vocab_size = len(tokenizer.word_index) + 1

# Convert questions to sequences and pad them
X = tokenizer.texts_to_sequences(data['Question'])
X = pad_sequences(X, padding='post')

# Tokenize the answers
answer_tokenizer = Tokenizer()
answer_tokenizer.fit_on_texts(data['Answer'])
answer_vocab_size = len(answer_tokenizer.word_index) + 1

# Convert answers to sequences and pad them
y = answer_tokenizer.texts_to_sequences(data['Answer'])
y = pad_sequences(y, padding='post')

# Filter out empty sequences
non_empty_indices = [i for i, seq in enumerate(y) if len(seq) > 0]
X = X[non_empty_indices]
y = np.array([seq[0] for seq in y if len(seq) > 0])

# Load GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Save the processed data for later use
np.save('X.npy', X)
np.save('y.npy', y)
np.save('vocab_size.npy', vocab_size)
np.save('answer_vocab_size.npy', answer_vocab_size)
np.save('embedding_matrix.npy', embedding_matrix)

# Save the tokenizers
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('answer_tokenizer.pickle', 'wb') as handle:
    pickle.dump(answer_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Data preprocessing complete.")

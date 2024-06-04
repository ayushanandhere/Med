import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load the processed data
X = np.load('X.npy')
y = np.load('y.npy')
vocab_size = np.load('vocab_size.npy').item()
answer_vocab_size = np.load('answer_vocab_size.npy').item()
embedding_matrix = np.load('embedding_matrix.npy')

# Define a model with bidirectional LSTM
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=X.shape[1], trainable=False),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(answer_vocab_size, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Add early stopping and model checkpoint to avoid overfitting and save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_chatbot_model.keras', save_best_only=True, monitor='val_loss')

# Train the model for 2 epochs
history = model.fit(X, y, epochs=3, batch_size=16, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Save the final model and training history
model.save('final_chatbot_model.keras')

import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

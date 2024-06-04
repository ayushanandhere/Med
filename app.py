import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the best model and tokenizers
model = load_model('best_chatbot_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('answer_tokenizer.pickle', 'rb') as handle:
    answer_tokenizer = pickle.load(handle)

def get_response(question):
    # Tokenize and pad the input question
    seq = tokenizer.texts_to_sequences([question])
    padded_seq = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
    # Predict the answer
    pred = model.predict(padded_seq)
    answer_index = np.argmax(pred, axis=1)[0]
    st.write(f"Debug: Predicted answer index: {answer_index}")  # Debug statement
    # Map the predicted index to the corresponding answer
    response = []
    for word, index in answer_tokenizer.word_index.items():
        if index == answer_index:
            response.append(word)
            break
    if response:
        return ' '.join(response)
    return "I don't understand the question."

# Streamlit UI
st.title('Medical Chatbot')
st.write("Ask the chatbot any medical question and it will try to answer based on the trained model.")
user_input = st.text_input('Ask me anything about medical advice:')
if st.button('Submit'):
    if user_input:
        response = get_response(user_input)
        st.write(f"Chatbot: {response}")

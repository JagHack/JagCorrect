import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout # Added Dropout

# Credited to JagHack

app = Flask(__name__)

# --- Copied from train_ai.py (with fixes) ---
def create_seq2seq_model(vocab_size, max_seq_len, embedding_dim=128, latent_dim=256):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_len,), dtype='int32')
    encoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_len,), dtype='int32')
    decoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dropout = Dropout(0.5)(decoder_outputs) # Use Dropout from keras.layers
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_dropout)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
# --- End of copy ---

# Global variables for models and vocab
encoder_model = None
decoder_model = None
vocab = None
id_to_char = None
max_seq_len = 100
latent_dim = 256 # Consistent with training

# Load models and vocabulary (in-memory creation)
try:
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    embedding_dim = 128 # Consistent with training

    print("Creating full model architecture for app...")
    full_model = create_seq2seq_model(vocab_size, max_seq_len, embedding_dim, latent_dim)

    print("Loading weights from seq2seq_model.h5 for app...")
    full_model.load_weights('seq2seq_model.h5')
    
    # Create inference models from the full model's layers
    print("Creating inference models from loaded weights for app...")
    encoder_inputs = full_model.input[0]
    _, state_h_enc, state_c_enc = full_model.layers[4].output # Encoder LSTM is layer 4
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_layer = full_model.layers[3] # Decoder Embedding is layer 3
    decoder_lstm_layer = full_model.layers[5]      # Decoder LSTM is layer 5
    decoder_dense_layer = full_model.layers[6]     # Decoder Dense is layer 6

    decoder_inputs_inf = Input(shape=(1,), dtype='int32')
    decoder_embedding_output_inf = decoder_embedding_layer(decoder_inputs_inf)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
        decoder_embedding_output_inf, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense_layer(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs_inf] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    print("Inference models created successfully for app.")

except Exception as e:
    print(f"Error loading models or vocabulary for app: {e}")
    print("Please ensure 'vocab.json' and 'seq2seq_model.h5' exist and are valid.")
    encoder_model = None
decoder_model = None
vocab = None

def tokenize_sentence(sentence, vocab, max_len):
    token_ids = [vocab.get('<SOS>')]
    for char in sentence:
        token_ids.append(vocab.get(char, vocab.get('<UNK>')))
    token_ids.append(vocab.get('<EOS>'))
    return tf.keras.preprocessing.sequence.pad_sequences([token_ids], maxlen=max_len, padding='post')

# Modified to accept models as arguments (though they are global here, it's good practice)
def decode_sequence(input_seq, encoder_model_arg, decoder_model_arg, vocab_arg, id_to_char_arg, max_seq_len_arg):
    # Encode the input as state vectors.
    states_value = encoder_model_arg.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab_arg['<SOS>']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model_arg.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id_to_char_arg.get(sampled_token_index, '<UNK>')

        # Exit condition: either hit max length or find stop character.
        if sampled_char == '<EOS>' or len(decoded_sentence) > max_seq_len_arg:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    # Use global models and vocab
    if encoder_model is None or decoder_model is None or vocab is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    input_sentence = data['sentence']
    input_seq = tokenize_sentence(input_sentence, vocab, max_seq_len)
    # Pass global models and vocab to decode_sequence
    corrected_sentence = decode_sequence(input_seq, encoder_model, decoder_model, vocab, id_to_char, max_seq_len)
    
    return jsonify({'corrected_sentence': corrected_sentence})

if __name__ == '__main__':
    app.run(debug=True)
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# --- Copied from train_ai.py ---
def create_seq2seq_model(vocab_size, max_seq_len, embedding_dim=128, latent_dim=256):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_len,), dtype='int32')
    encoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_len,), dtype='int32')
    decoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
# --- End of copy ---

def tokenize_sentence(sentence, vocab, max_len):
    token_ids = [vocab.get('<SOS>')]
    for char in sentence:
        token_ids.append(vocab.get(char, vocab.get('<UNK>')))
    token_ids.append(vocab.get('<EOS>'))
    return tf.keras.preprocessing.sequence.pad_sequences([token_ids], maxlen=max_len, padding='post')

# Modified to accept models as arguments
def decode_sequence(input_seq, encoder_model, decoder_model, vocab, id_to_char, max_seq_len):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab['<SOS>']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id_to_char.get(sampled_token_index, '<UNK>')

        # Exit condition: either hit max length or find stop character.
        if sampled_char == '<EOS>' or len(decoded_sentence) > max_seq_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

if __name__ == '__main__':
    try:
        with open('vocab.json', 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print("Error: vocab.json not found. Please ensure the file exists.")
        exit()

    id_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    max_seq_len = 100
    embedding_dim = 128
    latent_dim = 256

    # 1. Recreate the main model architecture
    print("Creating full model architecture...")
    full_model = create_seq2seq_model(vocab_size, max_seq_len, embedding_dim, latent_dim)

    # 2. Load the weights
    try:
        print("Loading weights from seq2seq_model.h5...")
        full_model.load_weights('seq2seq_model.h5')
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Please ensure 'seq2seq_model.h5' exists and is a valid weights file.")
        exit()
    
    # 3. Create inference models from the full model's layers
    print("Creating inference models from loaded weights...")
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
    
    print("Inference models created successfully.")

    # 4. Test with a sample sentence
    input_sentence = "this is a smaple snetence"
    input_seq = tokenize_sentence(input_sentence, vocab, max_seq_len)
    
    # Pass models and other necessary args to the function
    corrected_sentence = decode_sequence(input_seq, encoder_model, decoder_model, vocab, id_to_char, max_seq_len)
    
    print(f"Original sentence: {input_sentence}")
    print(f"Corrected sentence: {corrected_sentence}")
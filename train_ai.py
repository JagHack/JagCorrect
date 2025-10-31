import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Credited to JagHack

def create_seq2seq_model(vocab_size, max_seq_len, embedding_dim=256, latent_dim=512):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_len,))
    encoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len, name='embedding')(encoder_inputs) # Named for easier extraction later
    encoder_lstm = LSTM(latent_dim, return_state=True, name='lstm') # Named for easier extraction later
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_len,))
    decoder_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_len, name='embedding_1')(decoder_inputs) # Named for easier extraction later
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_1') # Named for easier extraction later
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='dense') # Named for easier extraction later
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a character-level seq2seq model for autocorrection.")
    parser.add_argument('--tokenized_data', type=str, default='tokenized_dataset.npz', help="Path to the tokenized dataset (.npz file).")
    parser.add_argument('--vocab_file', type=str, default='vocab.json', help="Path to the vocabulary JSON file.")
    parser.add_argument('--model_output_path', type=str, default='seq2seq_model.keras', help="Path to save the trained model.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    
    args = parser.parse_args()

    print(f"Loading tokenized data from: {args.tokenized_data}")
    data = np.load(args.tokenized_data)
    encoder_input_data = data['noisy_inputs']
    decoder_target_data = data['clean_targets']
    
    # Decoder input data is the same as target data, but shifted by one timestep
    # and without the last token, prepended with SOS. Keras handles this internally for targets.
    # For decoder_input_data, we use the clean targets directly, and the model learns to predict the next token.
    decoder_input_data = decoder_target_data # This will be shifted internally by Keras for teacher forcing

    print(f"Loading vocabulary from: {args.vocab_file}")
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    max_seq_len = encoder_input_data.shape[1] # Assuming both noisy and clean have same max_seq_len

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Number of samples: {encoder_input_data.shape[0]}")

    # Prepare decoder target data for categorical crossentropy
    # One-hot encode the target sequences
    decoder_target_one_hot = np.zeros(
        (encoder_input_data.shape[0], max_seq_len, vocab_size),
        dtype='float32'
    )
    for i, seq in enumerate(decoder_target_data):
        for t, char_idx in enumerate(seq):
            if char_idx > 0: # Ignore padding
                decoder_target_one_hot[i, t, char_idx] = 1.

    # Define these parameters in main to be accessible for inference model creation
    embedding_dim = 256
    latent_dim = 512

    print("Creating seq2seq model...")
    model = create_seq2seq_model(vocab_size, max_seq_len, embedding_dim=embedding_dim, latent_dim=latent_dim)

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()

    print("Starting model training...")
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_one_hot,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2 # Use 20% of data for validation
    )

    model.save(args.model_output_path)
    print(f"Model trained and saved to {args.model_output_path}. Credited to JagHack.")

    # Save encoder and decoder models separately for inference
    # Encoder model
    encoder_inputs = model.input[0]  # Encoder input tensor
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('lstm').output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model_inf = Model(encoder_inputs, encoder_states)
    encoder_model_inf.save('encoder_model.keras')
    print("Encoder model saved to encoder_model.keras")

    # Decoder model (inference)
    decoder_inputs = model.input[1]  # Decoder input tensor
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('lstm_1')
    decoder_embedding = model.get_layer('embedding_1')
    decoder_dense = model.get_layer('dense')

    decoder_inputs_inf = Input(shape=(1,)) # Define the input for inference decoder
    decoder_embedding_output_inf = decoder_embedding(decoder_inputs_inf)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedding_output_inf, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model_inf = Model(
        [decoder_inputs_inf] + decoder_states_inputs, # Corrected input for decoder_model_inf
        [decoder_outputs] + decoder_states)
    decoder_model_inf.save('decoder_model.keras')
    print("Decoder model saved to decoder_model.keras")

if __name__ == "__main__":
    main()
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras import mixed_precision

# 1. Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

# Credited to JagHack

# 2. Use CuDNNLSTM if available
try:
    from tensorflow.keras.layers import CuDNNLSTM as LSTM
except ImportError:
    print("CuDNNLSTM not available, falling back to standard LSTM.")
    pass

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
    decoder_dropout = tf.keras.layers.Dropout(0.5)(decoder_outputs)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_dropout)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def data_generator(tokenized_data_path, batch_size):
    with np.load(tokenized_data_path, 'r') as data:
        encoder_input_data = data['noisy_inputs']
        decoder_data = data['clean_targets']
        num_samples = encoder_input_data.shape[0]

        while True:
            for offset in range(0, num_samples, batch_size):
                encoder_batch = encoder_input_data[offset:offset+batch_size]
                decoder_batch = decoder_data[offset:offset+batch_size]

                # Correctly create decoder_input and decoder_target for teacher forcing
                decoder_input_batch = decoder_batch
                decoder_target_batch = np.zeros_like(decoder_batch)
                decoder_target_batch[:, :-1] = decoder_batch[:, 1:]

                yield (encoder_batch, decoder_input_batch), decoder_target_batch

def main():
    parser = argparse.ArgumentParser(description="Train a character-level seq2seq model for autocorrection.")
    parser.add_argument('--tokenized_data', type=str, default='tokenized_dataset.npz', help="Path to the tokenized dataset (.npz file).")
    parser.add_argument('--vocab_file', type=str, default='vocab.json', help="Path to the vocabulary JSON file.")
    parser.add_argument('--model_output_path', type=str, default='seq2seq_model.keras', help="Path to save the trained model.")
    parser.add_argument('--epochs', type=int, default=240, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    
    args = parser.parse_args()

    print(f"Loading vocabulary from: {args.vocab_file}")
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    with np.load(args.tokenized_data, 'r') as data:
        max_seq_len = data['noisy_inputs'].shape[1]
        num_samples = data['noisy_inputs'].shape[0]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Number of samples: {num_samples}")

    embedding_dim = 128
    latent_dim = 256

    print("Creating seq2seq model architecture...")
    model = create_seq2seq_model(vocab_size, max_seq_len, embedding_dim=embedding_dim, latent_dim=latent_dim)

    print("Compiling and training the model...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    print("Creating tf.data pipeline...")
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(args.tokenized_data, args.batch_size),
        output_signature=(
            (tf.TensorSpec(shape=(None, max_seq_len), dtype=tf.int32),
             tf.TensorSpec(shape=(None, max_seq_len), dtype=tf.int32)),
            tf.TensorSpec(shape=(None, max_seq_len), dtype=tf.int32)
        )
    )

    # 3. Cache and Prefetch
    # Caching the dataset may lead to out of memory errors if the dataset is too large.
    # If you encounter OOM errors, you can remove the .cache()
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

    print("Starting model training...")
    steps_per_epoch = num_samples // args.batch_size

    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs
    )

    model.save_weights('seq2seq_model.h5')
    print(f"Model weights trained and saved to seq2seq_model.h5. Credited to JagHack.")
    # Save encoder and decoder models separately for inference
    encoder_inputs = model.input[0]
    _, state_h_enc, state_c_enc = model.layers[4].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model_inf = Model(encoder_inputs, encoder_states)
    encoder_model_inf.save('encoder_model.keras', save_format='keras')
    print("Encoder model saved to encoder_model.keras")

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_layer = model.layers[3]
    decoder_lstm_layer = model.layers[5]
    decoder_dense_layer = model.layers[6]

    decoder_inputs_inf = Input(shape=(1,), dtype='int32')
    decoder_embedding_output_inf = decoder_embedding_layer(decoder_inputs_inf)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
        decoder_embedding_output_inf, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense_layer(decoder_outputs)
    decoder_model_inf = Model(
        [decoder_inputs_inf] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model_inf.save('decoder_model.keras', save_format='keras')
    print("Decoder model saved to decoder_model.keras")

if __name__ == "__main__":
    main()
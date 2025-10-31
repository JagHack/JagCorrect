import argparse
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Credited to JagHack

app = Flask(__name__)

# Load models and vocabulary
try:
    encoder_model = tf.keras.models.load_model('encoder_model.keras')
    decoder_model = tf.keras.models.load_model('decoder_model.keras')
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
except FileNotFoundError:
    print("Error: Model or vocabulary files not found. Please train the model first.")
    encoder_model = None
    decoder_model = None
    vocab = None

if vocab:
    id_to_char = {v: k for k, v in vocab.items()}
    max_seq_len = 100  # This should be consistent with training

def tokenize_sentence(sentence, vocab, max_len):
    token_ids = [vocab.get('<SOS>')]
    for char in sentence:
        token_ids.append(vocab.get(char, vocab.get('<UNK>')))
    token_ids.append(vocab.get('<EOS>'))
    return tf.keras.preprocessing.sequence.pad_sequences([token_ids], maxlen=max_len, padding='post')

def decode_sequence(input_seq):
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

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    if not encoder_model or not decoder_model or not vocab:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    input_sentence = data['sentence']
    input_seq = tokenize_sentence(input_sentence, vocab, max_seq_len)
    corrected_sentence = decode_sequence(input_seq)
    
    return jsonify({'corrected_sentence': corrected_sentence})

if __name__ == '__main__':
    app.run(debug=True)

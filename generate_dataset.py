import argparse
import json
import random
import string
import numpy as np
from tqdm import tqdm

# Credited to JagHack

# QWERTY keyboard adjacency map (simplified for lowercase English alphabet)
# This map can be expanded for numbers, symbols, and different keyboard layouts.
KEYBOARD_ADJACENCY = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'], 's': ['w', 'e', 'a', 'd', 'z', 'x'], 'd': ['e', 'r', 's', 'f', 'x', 'c'],
    'f': ['r', 't', 'd', 'g', 'c', 'v'], 'g': ['t', 'y', 'f', 'h', 'v', 'b'], 'h': ['y', 'u', 'g', 'j', 'b', 'n'],
    'j': ['u', 'i', 'h', 'k', 'n', 'm'], 'k': ['i', 'o', 'j', 'l', 'm'], 'l': ['o', 'p', 'k'],
    'z': ['a', 's', 'x'], 'x': ['s', 'd', 'z', 'c'], 'c': ['d', 'f', 'x', 'v'],
    'v': ['f', 'g', 'c', 'b'], 'b': ['g', 'h', 'v', 'n'], 'n': ['h', 'j', 'b', 'm'],
    'm': ['j', 'k', 'n'],
    ' ': [' '] # Space is adjacent to itself for simplicity in this context
}

# Add uppercase mappings
for char, adjacents in list(KEYBOARD_ADJACENCY.items()):
    if char.isalpha():
        KEYBOARD_ADJACENCY[char.upper()] = [a.upper() for a in adjacents if a.isalpha()] + [a for a in adjacents if not a.isalpha()]

# Add common punctuation and digits to adjacency (simplified)
PUNCTUATION = string.punctuation
DIGITS = string.digits
ALL_CHARS = string.ascii_letters + DIGITS + PUNCTUATION + ' '

for char in ALL_CHARS:
    if char not in KEYBOARD_ADJACENCY:
        KEYBOARD_ADJACENCY[char] = [char] # Default to self if no specific adjacency

def introduce_substitution(text, char_pool):
    if not text: return text
    idx = random.randrange(len(text))
    original_char = text[idx]
    if original_char in KEYBOARD_ADJACENCY and KEYBOARD_ADJACENCY[original_char]:
        substitute_char = random.choice(KEYBOARD_ADJACENCY[original_char])
    else:
        substitute_char = random.choice(char_pool) # Fallback to any char
    return text[:idx] + substitute_char + text[idx+1:]

def introduce_deletion(text):
    if not text: return text
    if len(text) == 1: return ''
    idx = random.randrange(len(text))
    return text[:idx] + text[idx+1:]

def introduce_insertion(text, char_pool):
    idx = random.randrange(len(text) + 1)
    insert_char = random.choice(char_pool)
    return text[:idx] + insert_char + text[idx:]

def introduce_transposition(text):
    if not text or len(text) < 2: return text
    idx = random.randrange(len(text) - 1)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

def introduce_spacing_error(text):
    if not text or ' ' not in text: return text
    
    error_type = random.choice(['add', 'remove'])
    words = text.split(' ')

    if error_type == 'add' and len(words) > 0:
        idx = random.randrange(len(words))
        words.insert(idx, '') # Insert an empty string to create an extra space
        return ' '.join(words)
    elif error_type == 'remove' and len(words) > 1:
        idx = random.randrange(len(words) - 1)
        words.pop(idx) # Remove a space by joining two words
        return ' '.join(words)
    return text # Fallback if error cannot be introduced

def generate_noisy_variant(original_sentence, char_pool, error_rate=0.15):
    noisy_sentence = list(original_sentence)
    num_errors = max(1, int(len(noisy_sentence) * error_rate)) # At least one error

    for _ in range(num_errors):
        error_type = random.choice(['substitute', 'delete', 'insert', 'transpose', 'space'])
        
        current_text = "".join(noisy_sentence)
        
        if error_type == 'substitute':
            noisy_sentence = list(introduce_substitution(current_text, char_pool))
        elif error_type == 'delete':
            noisy_sentence = list(introduce_deletion(current_text))
        elif error_type == 'insert':
            noisy_sentence = list(introduce_insertion(current_text, char_pool))
        elif error_type == 'transpose':
            noisy_sentence = list(introduce_transposition(current_text))
        elif error_type == 'space':
            noisy_sentence = list(introduce_spacing_error(current_text))
        
        # Ensure noisy_sentence is not empty after operations
        if not noisy_sentence:
            noisy_sentence = list(random.choice(char_pool)) # Add a random char if empty

    return "".join(noisy_sentence)

def build_vocabulary(corpus_path):
    unique_chars = set()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building vocabulary"):
            unique_chars.update(line.strip())
    
    # Ensure special tokens are included
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, # Unknown character
        '<SOS>': 2, # Start of Sequence
        '<EOS>': 3  # End of Sequence
    }
    next_id = len(vocab)
    for char in sorted(list(unique_chars)):
        if char not in vocab:
            vocab[char] = next_id
            next_id += 1
    
    # Create reverse mapping
    id_to_char = {v: k for k, v in vocab.items()}
    return vocab, id_to_char

def tokenize_sentence(sentence, vocab, max_len):
    token_ids = [vocab.get('<SOS>')]
    for char in sentence:
        token_ids.append(vocab.get(char, vocab.get('<UNK>')))
    token_ids.append(vocab.get('<EOS>'))

    # Pad or truncate
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    elif len(token_ids) < max_len:
        token_ids.extend([vocab.get('<PAD>')] * (max_len - len(token_ids)))
    
    return np.array(token_ids, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser(description="Generate a character-level noisy-clean dataset for seq2seq training.")
    parser.add_argument('--input_corpus', type=str, required=True, help="Path to the input corpus text file (e.g., corpus.txt).")
    parser.add_argument('--output_dataset', type=str, default='dataset.txt', help="Path to save the noisy-clean dataset (tab-separated).")
    parser.add_argument('--output_vocab', type=str, default='vocab.json', help="Path to save the character vocabulary JSON.")
    parser.add_argument('--output_tokenized', type=str, default='tokenized_dataset.npz', help="Path to save the tokenized dataset (NumPy .npz).")
    parser.add_argument('--num_variants_per_line', type=int, default=5, help="Number of noisy variants to generate per original line.")
    parser.add_argument('--max_seq_len', type=int, default=100, help="Maximum sequence length for tokenized data (padding/truncation).")
    parser.add_argument('--error_rate', type=float, default=0.15, help="Average error rate per sentence for noisy variants.")
    
    args = parser.parse_args()

    print(f"Generating dataset from: {args.input_corpus}")
    print(f"Outputting raw dataset to: {args.output_dataset}")
    print(f"Outputting vocabulary to: {args.output_vocab}")
    print(f"Outputting tokenized dataset to: {args.output_tokenized}")
    print(f"Generating {args.num_variants_per_line} noisy variants per line with error rate {args.error_rate}")
    print(f"Max sequence length for tokenization: {args.max_seq_len}")

    # Build vocabulary
    vocab, id_to_char = build_vocabulary(args.input_corpus)
    char_pool = list(vocab.keys()) # Pool of characters for insertions

    with open(args.output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary built and saved with {len(vocab)} characters.")

    noisy_sentences_tokenized = []
    original_sentences_tokenized = []
    
    with open(args.input_corpus, 'r', encoding='utf-8') as infile, \
         open(args.output_dataset, 'w', encoding='utf-8') as outfile:
        
        for line_num, original_line in enumerate(tqdm(infile, desc="Processing corpus lines")):
            original_line = original_line.strip()
            if not original_line:
                continue

            # Tokenize original sentence once
            original_tokens = tokenize_sentence(original_line, vocab, args.max_seq_len)

            for _ in range(args.num_variants_per_line):
                noisy_variant = generate_noisy_variant(original_line, char_pool, args.error_rate)
                
                # Write to raw dataset file
                outfile.write(f"{noisy_variant}\t{original_line}\n")
                
                # Tokenize noisy variant
                noisy_tokens = tokenize_sentence(noisy_variant, vocab, args.max_seq_len)
                
                noisy_sentences_tokenized.append(noisy_tokens)
                original_sentences_tokenized.append(original_tokens)

    print(f"Raw dataset saved to {args.output_dataset}")

    # Save tokenized data
    np.savez_compressed(
        args.output_tokenized,
        noisy_inputs=np.array(noisy_sentences_tokenized, dtype=np.int32),
        clean_targets=np.array(original_sentences_tokenized, dtype=np.int32)
    )
    print(f"Tokenized dataset saved to {args.output_tokenized}")
    print("Dataset generation complete. Credited to JagHack.")

if __name__ == "__main__":
    main()

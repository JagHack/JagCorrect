import argparse
import json
import numpy as np
from tqdm import tqdm

def build_vocabulary(sentences):
    unique_chars = set()
    for sentence in tqdm(sentences, desc="Building vocabulary"):
        unique_chars.update(sentence.strip())
    
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
    
    id_to_char = {v: k for k, v in vocab.items()}
    return vocab, id_to_char

def tokenize_sentence(sentence, vocab, max_len):
    token_ids = [vocab.get('<SOS>')]
    for char in sentence:
        token_ids.append(vocab.get(char, vocab.get('<UNK>')))
    token_ids.append(vocab.get('<EOS>'))

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    elif len(token_ids) < max_len:
        token_ids.extend([vocab.get('<PAD>')] * (max_len - len(token_ids)))
    
    return np.array(token_ids, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser(description="Generate a character-level noisy-clean dataset from the GitHub Typo Corpus.")
    parser.add_argument('--input_jsonl', type=str, default='github-typo-corpus.v1.0.0.jsonl', help="Path to the input JSONL file.")
    parser.add_argument('--output_vocab', type=str, default='vocab.json', help="Path to save the character vocabulary JSON.")
    parser.add_argument('--output_tokenized', type=str, default='tokenized_dataset.npz', help="Path to save the tokenized dataset (NumPy .npz).")
    parser.add_argument('--max_seq_len', type=int, default=100, help="Maximum sequence length for tokenized data (padding/truncation).")
    
    args = parser.parse_args()

    print(f"Generating dataset from: {args.input_jsonl}")
    print(f"Outputting vocabulary to: {args.output_vocab}")
    print(f"Outputting tokenized dataset to: {args.output_tokenized}")
    print(f"Max sequence length for tokenization: {args.max_seq_len}")

    noisy_sentences = []
    clean_sentences = []

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL file"):
            data = json.loads(line)
            for edit in data.get('edits', []):
                if edit.get('is_typo') and edit.get('src') and edit.get('tgt'):
                    noisy_sentences.append(edit['src']['text'])
                    clean_sentences.append(edit['tgt']['text'])

    print(f"Found {len(noisy_sentences)} typo edits.")

    # Build vocabulary from both noisy and clean sentences
    all_sentences = noisy_sentences + clean_sentences
    vocab, id_to_char = build_vocabulary(all_sentences)

    with open(args.output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary built and saved with {len(vocab)} characters.")

    noisy_sentences_tokenized = []
    clean_sentences_tokenized = []

    for noisy_sentence in tqdm(noisy_sentences, desc="Tokenizing noisy sentences"):
        noisy_tokens = tokenize_sentence(noisy_sentence, vocab, args.max_seq_len)
        noisy_sentences_tokenized.append(noisy_tokens)

    for clean_sentence in tqdm(clean_sentences, desc="Tokenizing clean sentences"):
        clean_tokens = tokenize_sentence(clean_sentence, vocab, args.max_seq_len)
        clean_sentences_tokenized.append(clean_tokens)

    np.savez_compressed(
        args.output_tokenized,
        noisy_inputs=np.array(noisy_sentences_tokenized, dtype=np.int32),
        clean_targets=np.array(clean_sentences_tokenized, dtype=np.int32)
    )
    print(f"Tokenized dataset saved to {args.output_tokenized}")
    print("Dataset generation complete. Credited to JagHack.")

if __name__ == "__main__":
    main()
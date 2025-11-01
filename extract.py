import gzip
import shutil

with gzip.open('github-typo-corpus.v1.0.0.jsonl.gz', 'rb') as f_in:
    with open('github-typo-corpus.v1.0.0.jsonl', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

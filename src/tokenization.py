import re
from collections import Counter

class SimpleSubwordTokenizer:
    """
    Simple character-level tokenizer with BPE-style merges.
    Fixed version that properly handles merge tokens.
    """
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", blank_token="<blank>"):
        self.vocab = vocab or {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        # Build reverse mapping if vocab provided
        if vocab:
            self.id2tok = {i: t for t, i in vocab.items()}
            # Build merge map: maps (id1, id2) -> merged_id
            self.merge_map = {}
            for tok, tok_id in vocab.items():
                if tok.startswith("<") and tok.endswith(">") and "_" in tok:
                    try:
                        parts = tok[1:-1].split("_")
                        if len(parts) == 2:
                            a_str, b_str = parts
                            self.merge_map[(int(a_str), int(b_str))] = tok_id
                    except ValueError:
                        pass  # Skip malformed merge tokens
        else:
            self.id2tok = {}
            self.merge_map = {}

    @staticmethod
    def _tokenize_chars(text):
        # Keep all characters including spaces and punctuation
        return list(text)

    def train(self, texts, vocab_size=2000, min_freq=2):
        # Start with special tokens and characters
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize_chars(t))

        vocab = {self.blank_token: 0, self.pad_token: 1, self.unk_token: 2}
        next_id = 3

        # Add frequent characters
        for ch, c in counter.most_common():
            if c >= min_freq and next_id < vocab_size:
                vocab[ch] = next_id
                next_id += 1

        # BPE-style merges: store as <id1_id2> format
        def encode_with_vocab(s, current_vocab):
            """Encode string using current vocab (no merges yet)"""
            return [current_vocab.get(ch, current_vocab[self.unk_token]) for ch in self._tokenize_chars(s)]

        # Perform merge iterations
        for merge_iter in range(10):  # More iterations for better merges
            if next_id >= vocab_size:
                break

            pair_counts = Counter()
            for t in texts:
                ids = encode_with_vocab(t, vocab)
                for i in range(len(ids) - 1):
                    pair_counts[(ids[i], ids[i+1])] += 1

            # Add most frequent pair as new token
            if not pair_counts:
                break

            (a, b), count = pair_counts.most_common(1)[0]
            if count < min_freq:
                break

            # Store merge as <id1_id2>
            merge_token = f"<{a}_{b}>"
            if merge_token not in vocab and next_id < vocab_size:
                vocab[merge_token] = next_id
                next_id += 1

        self.vocab = vocab
        self.id2tok = {i: t for t, i in vocab.items()}

        # Build merge map for efficient encoding
        self.merge_map = {}
        for tok, tok_id in vocab.items():
            if tok.startswith("<") and tok.endswith(">") and "_" in tok:
                try:
                    parts = tok[1:-1].split("_")
                    if len(parts) == 2:
                        a_str, b_str = parts
                        self.merge_map[(int(a_str), int(b_str))] = tok_id
                except ValueError:
                    pass

    def encode(self, text):
        """Encode text to list of token IDs"""
        if not self.vocab:
            raise RuntimeError("Tokenizer not trained")

        # First pass: convert to character IDs
        chars = self._tokenize_chars(text)
        ids = [self.vocab.get(ch, self.vocab[self.unk_token]) for ch in chars]

        # Second pass: apply merges greedily
        changed = True
        while changed:
            changed = False
            i = 0
            new_ids = []
            while i < len(ids):
                # Try to merge current and next token
                if i + 1 < len(ids):
                    pair = (ids[i], ids[i+1])
                    if pair in self.merge_map:
                        new_ids.append(self.merge_map[pair])
                        i += 2
                        changed = True
                        continue
                new_ids.append(ids[i])
                i += 1
            ids = new_ids

        return ids

    def decode(self, ids):
        """Decode list of token IDs back to text"""
        if not self.id2tok:
            raise RuntimeError("Tokenizer not trained")

        # Recursively expand merge tokens
        def expand_token(tok_id):
            tok = self.id2tok.get(int(tok_id), self.unk_token)

            # If it's a merge token, recursively expand
            if tok.startswith("<") and tok.endswith(">") and "_" in tok:
                try:
                    parts = tok[1:-1].split("_")
                    if len(parts) == 2:
                        a_str, b_str = parts
                        return expand_token(int(a_str)) + expand_token(int(b_str))
                except ValueError:
                    pass

            # Skip special tokens
            if tok in {self.blank_token, self.pad_token, self.unk_token}:
                return ""

            return tok

        return "".join(expand_token(i) for i in ids)
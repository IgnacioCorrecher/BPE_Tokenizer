class BytePairEncoding:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def get_stats(self, ids, counts=None) -> dict:
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx) -> dict:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes)

        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges 
        self.vocab = vocab   

    def encode(self, text) -> list:
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs = self.get_stats(tokens).keys()
            min_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if min_pair not in self.merges:
                break  # nothing more to merge
            idx = self.merges[min_pair]
            tokens = self.merge(tokens, min_pair, idx)
        return tokens

    def decode(self, ids) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

if __name__ == "__main__":
    text = "Hello World how are you doing?"
    bpe = BytePairEncoding()
    
    # Training process
    bpe.train(text, 300)
    
    # Encoding process
    encoded_ids = bpe.encode(text)
    print(f"Encoded IDs: {encoded_ids}")
    
    # Decoding process
    decoded_text = bpe.decode(encoded_ids)
    print(f"Decoded text: {decoded_text}")

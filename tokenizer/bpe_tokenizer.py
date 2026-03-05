class BPETokenizer:

    def __init__(self, merges):
        self.merges = merges
        self.token_to_id = {}
        self.id_to_token = {}

    def encode_word(self, word):
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == pair:
                    tokens[i:i+2] = ["".join(pair)]
                else:
                    i += 1
        return tokens

    def encode(self, text):
        output_tokens = []
        words = text.split()
        for word in words:
            word_tokens = self.encode_word(word)
            output_tokens.extend(word_tokens)
        return output_tokens

    def decode(self, tokens):
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
    
    def build_vocab(self, tokens):
        unique_tokens = sorted(set(tokens))
        for idx, token in enumerate(unique_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode_to_ids(self, text):
        tokens = self.encode(text)
        ids = [self.token_to_id[token] for token in tokens]
        return ids
    
    def decode_from_ids(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        return self.decode(tokens)
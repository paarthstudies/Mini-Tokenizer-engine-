from collections import Counter


def get_vocab(text):
    """
    Build initial vocabulary where each word
    is split into characters.
    """

    vocab = Counter()

    for word in text.split():
        tokens = " ".join(list(word)) + " </w>"
        vocab[tokens] += 1

    return vocab


def get_pair_frequencies(vocab):
    """
    Count frequency of adjacent token pairs.
    """

    pairs = Counter()

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq

    return pairs


def merge_vocab(pair, vocab):
    """
    Merge the most frequent pair.
    """

    new_vocab = {}

    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab


def train_bpe(text, num_merges=50):
    """
    Learn BPE merges from the corpus.
    """

    vocab = get_vocab(text)

    merges = []

    for _ in range(num_merges):

        pairs = get_pair_frequencies(vocab)

        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)

        vocab = merge_vocab(best_pair, vocab)

        merges.append(best_pair)

    return merges
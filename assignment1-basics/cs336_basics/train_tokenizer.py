import regex as re
INITIAL_VOCAB_SIZE = 256

def initialize_byte_vocab(vocab: dict) -> dict[int, bytes]:

    for tokenID in range(INITIAL_VOCAB_SIZE):
        vocab[tokenID] = bytes([tokenID])

    return vocab


def add_special_tokens(vocab: dict) -> dict[int, bytes]:

    vocab.update({INITIAL_VOCAB_SIZE: "<|endoftext|>".encode("utf-8")})

    return vocab


def initialize_vocab(vocab: dict, special_tokens: list) -> dict[int, bytes]:

    initialize_byte_vocab(vocab)

    if special_tokens == None:
        return vocab

    else:
        num_special_tokens = len(special_tokens)
        for iteration in range(num_special_tokens):
            vocab.update({INITIAL_VOCAB_SIZE + iteration: special_tokens[iteration].encode("utf-8")})

        return vocab


    
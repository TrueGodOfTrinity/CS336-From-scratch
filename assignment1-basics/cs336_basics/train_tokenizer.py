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

def simple_pre_tokenization(chunk: str, special_tokens: list) -> dict[str, int]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    chunk_list = [chunk]
    
    for iteration in range(len(special_tokens)):
        chunk_tmp = list()
        for chunk in chunk_list:
            chunk = chunk.split(special_tokens[iteration])
            chunk_tmp.extend(chunk)
        chunk_list = chunk_tmp
    
    pre_token_count = dict()

    for sub_chunk in chunk_list:

        pre_tokens = re.findall(PAT,sub_chunk)

        for pre_token in pre_tokens:
                
            if pre_token not in pre_token_count.keys():
                pre_token_count[pre_token] = 1
            else:
                pre_token_count[pre_token] += 1

    return pre_token_count


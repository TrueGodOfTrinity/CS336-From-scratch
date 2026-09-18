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
        for index in range(num_special_tokens):
            vocab.update({INITIAL_VOCAB_SIZE + index: special_tokens[index].encode("utf-8")})

        return vocab


def simple_pre_tokenization(chunk: str, special_tokens: list) -> dict[str, int]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    chunk_list = [chunk]
    
    for index in range(len(special_tokens)):
        chunk_tmp = list()
        for chunk in chunk_list:
            chunk = chunk.split(special_tokens[index])
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


def initialize_token_sequences(pre_token_count: dict) ->tuple[dict[tuple[bytes, ...], int], tuple[bytes, ...]]:

    encoded_pre_token_count = dict()
    for pre_token in pre_token_count.keys():
        encoded_pre_token_count.update({pre_token.encode("utf-8"): pre_token_count[pre_token]})

    token_sequence = tuple()
    token_sequences_count = dict()

    for encoded_pre_token in encoded_pre_token_count.keys():
        encoded_pre_token_list = list(encoded_pre_token)
        
        for index in range(len(encoded_pre_token_list)):
            encoded_pre_token_list[index] = bytes([encoded_pre_token_list[index]])

        token_sequence = tuple(encoded_pre_token_list)
        token_sequences_count.update({token_sequence: encoded_pre_token_count[encoded_pre_token]})

    return token_sequences_count, token_sequence


def count_adjacent_token_pairs(token_sequences_count: dict) -> dict[tuple[bytes, bytes], int]:

    adj_token_pairs_count = dict()

    for token_sequence in token_sequences_count.keys():

        for index in range(len(token_sequence) - 1):
            byte_pairs = (token_sequence[index], token_sequence[index + 1])
            if (byte_pairs) not in adj_token_pairs_count.keys():
                adj_token_pairs_count[byte_pairs] = token_sequences_count[token_sequence]
            else:
                adj_token_pairs_count[byte_pairs] += token_sequences_count[token_sequence]

    return adj_token_pairs_count


def select_best_pair(adj_token_pairs_count: dict) -> tuple[bytes, bytes] | None:

    try:
        max_frequency = max(adj_token_pairs_count.values())
        pairs = [pairs for pairs, frequency in adj_token_pairs_count.items() if frequency == max_frequency]
    except ValueError:
        return None 
    
    best_pair = max(pairs)

    return best_pair


def merge_token_pair(token_sequence: tuple, best_pair: tuple) -> tuple[bytes, ...]:

    merged_token_pairs = list()

    index = 0

    while(index< len(token_sequence)):
        if index + 1 < len(token_sequence) and (token_sequence[index], token_sequence[index + 1]) == best_pair:
            merged_token_pair = token_sequence[index] + token_sequence[index + 1]
            merged_token_pairs.append(merged_token_pair)
            index += 2

        else:
            merged_token_pairs.append(token_sequence[index])
            index += 1

    merged_token_pairs = tuple(merged_token_pairs) 

    return merged_token_pairs

        
def merge_token_sequences(token_sequences_count: dict, best_pair: tuple) -> dict[tuple[bytes, ...], int]:

    merged_token_sequences = dict()

    for token_sequence in token_sequences_count.keys():

        merged_token_pairs = merge_token_pair(token_sequence, best_pair)

        if merged_token_pairs not in merged_token_sequences.keys():
            merged_token_sequences.update({merged_token_pairs: token_sequences_count[token_sequence]})
        else:
            merged_token_sequences[merged_token_pairs] += token_sequences_count[token_sequence]

    return merged_token_sequences


def add_merged_token_into_vocab(merges: list, vocab: dict, best_pair: tuple) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    best_token = best_pair[0] + best_pair[1]
    vocab.update({len(vocab): best_token})

    merges.append(best_pair)


    return vocab, merges


import regex as re
import os
INITIAL_VOCAB_SIZE = 256

def initialize_byte_vocab(vocab: dict) -> dict[int, bytes]:

    for tokenID in range(INITIAL_VOCAB_SIZE):
        vocab[tokenID] = bytes([tokenID])

    return vocab


def add_special_tokens(vocab: dict) -> dict[int, bytes]:

    vocab.update({INITIAL_VOCAB_SIZE: "<|endoftext|>".encode("utf-8")})

    return vocab


def initialize_vocab(vocab: dict, special_tokens: list) -> dict[int, bytes]:

    """Initialize the vocabulary with all 256 byte values and any special tokens."""

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


def initialize_token_sequences(pre_token_count: dict) ->dict[tuple[bytes, ...], int]:

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

    return token_sequences_count


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


def count_pair_contributions(token_sequence:tuple, sequence_frequency: int) -> dict[tuple[bytes, bytes], int]:

    pair_contributions = dict()

    for index in range(len(token_sequence) - 1):
        byte_pairs = (token_sequence[index], token_sequence[index + 1])
        if (byte_pairs) not in pair_contributions:
            pair_contributions[byte_pairs] = sequence_frequency
        else:
            pair_contributions[byte_pairs] += sequence_frequency

    return pair_contributions


def build_pair_index(sequences: list) -> dict[tuple[bytes, bytes], set[int]]:

    pair_index = dict()
    
    for token_sequence_id, token_sequence in enumerate(sequences):
        for index in range(len(token_sequence) - 1):
            byte_pair = (token_sequence[index], token_sequence[index + 1])
            if byte_pair not in pair_index.keys():
                token_sequence_ids = set()
                token_sequence_ids.add(token_sequence_id)
                pair_index[byte_pair] = token_sequence_ids
            else:
                pair_index[byte_pair].add(token_sequence_id)
                
    return pair_index


def select_best_pair(adj_token_pairs_count: dict) -> tuple[bytes, bytes] | None:

    try:
        max_sequence_frequency = max(adj_token_pairs_count.values())
        pairs = [pairs for pairs, sequence_frequency in adj_token_pairs_count.items() if sequence_frequency == max_sequence_frequency]
    except ValueError:
        return None 
    
    best_pair = max(pairs)

    return best_pair


def merge_token_pair(token_sequence: tuple, best_pair: tuple) -> tuple[bytes, ...]:

    """Merge non-overlapping occurrences of the selected pair from left to right."""

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

        
def merge_token_sequences(sequences: list, sequence_ids: set, best_pair: tuple):

    """Merge the selected pair in all token sequences and aggregate their counts."""
    # brand new version: aiming to decrease computational cost

    for token_sequence_id in sequence_ids:
        merged_token_pair = merge_token_pair(sequences[token_sequence_id], best_pair)
        sequences[token_sequence_id] = merged_token_pair

    return None


def add_merged_token_into_vocab(merges: list, vocab: dict, best_pair: tuple) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    best_token = best_pair[0] + best_pair[1]
    vocab.update({len(vocab): best_token})

    merges.append(best_pair)

    return vocab, merges


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) ->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 

    vocab = dict()
    merges = list()
    vocab = initialize_vocab(vocab, special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    token_sequences_count = initialize_token_sequences(simple_pre_tokenization(corpus, special_tokens))

    sequences = list()
    sequence_frequencies = list()
    for token_sequence, sequence_frequency in token_sequences_count.items():
        sequences.append(token_sequence)
        sequence_frequencies.append(sequence_frequency)
    
    # Build the reverse index once: each pair points to the sequence IDs that contain it.
    pair_index = build_pair_index(sequences)

    pair_counts = dict()
    # Initialize the global pair counts from every sequence and its corpus frequency.
    for index in range(len(sequences)):
        pair_contributions = count_pair_contributions(sequences[index], sequence_frequencies[index])
        for pair, count in pair_contributions.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    while(len(vocab) < vocab_size):       
        # Select the most frequent pair; select_best_pair also applies the tie-breaking rule.
        best_pair = select_best_pair(pair_counts)
       
        if best_pair == None:
            break
        else:
            # Copy the affected IDs because the index sets are updated below.
            token_sequence_ids = pair_index[best_pair]
            
            temp_token_seq_ids = token_sequence_ids.copy()
            # Remove old pair-index entries for the sequences that will change.
            for index in temp_token_seq_ids:
                pairs_list = list()
                pairs_list = count_pair_contributions(sequences[index], sequence_frequencies[index]).keys()

                for pair in pairs_list:
                    pair_index[pair].remove(index)
                    if pair_index[pair] == set():
                        del pair_index[pair]

            # Remove the old pair-count contributions of the affected sequences.
            old_contributions = dict()
            for index in temp_token_seq_ids:
                pair_contributions = count_pair_contributions(sequences[index], sequence_frequencies[index])
                for pair, count in pair_contributions.items():
                    old_contributions[pair] = old_contributions.get(pair, 0) + count
            for pair in old_contributions:
                pair_counts[pair] -= old_contributions[pair]            
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                       
            # Merge only the affected sequences; all other sequences remain unchanged.
            merge_token_sequences(sequences, temp_token_seq_ids, best_pair)

            # Add the new pair counts and pair-index entries for the updated sequences.
            for index in temp_token_seq_ids: 
                pair_contributions = count_pair_contributions(sequences[index], sequence_frequencies[index])
                for pair, count in pair_contributions.items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

                    if pair not in pair_index:
                        pair_index[pair] = set()
                        pair_index[pair].add(index)
                    else:
                        pair_index[pair].add(index)            

            # Record the newly created token and the merge operation.
            vocab, merges = add_merged_token_into_vocab(merges, vocab, best_pair)

    return vocab, merges

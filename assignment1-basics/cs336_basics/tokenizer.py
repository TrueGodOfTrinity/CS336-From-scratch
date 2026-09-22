import regex as re
import json
from collections.abc import Iterable, Iterator
from cs336_basics import train_tokenizer


class Tokenizer():


    def __init__(self, vocab: dict, merges: list, special_tokens = None) -> None:

        self.vocab = vocab.copy() # 构造 tokenizer 后，外部字典也新增了特殊 token, 保存词表副本，让不同实例互不影响
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens is None:
            self.special_tokens = []
            
        self.reversed_vocab = {token_byte: token_id for token_id, token_byte in self.vocab.items()}
        self.merge_rank = {merge: rank for rank, merge in enumerate(self.merges)}

        added_sepecial_token_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:

            special_token_byte = special_token.encode("utf-8")
            if special_token_byte not in self.reversed_vocab:
                self.reversed_vocab[special_token_byte] = added_sepecial_token_id
                self.vocab[added_sepecial_token_id] = special_token_byte
        

    def decode(self,token_ids: list) -> str:
        # tokenID -> text for reading
        token_bytes= list()
        for token_id in token_ids:
            token_byte = self.vocab[token_id]
            token_bytes.append(token_byte)
        
        result = b"".join(token_bytes).decode("utf-8", errors="replace")

        return result
            
        
    def _encode_ordinary(self, text: str) -> list[int]:
        # encode the text without special tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        result = list()
        
        chunk_list = [text]
        pre_tokens = list()


        for sub_chunk in chunk_list:
            sub_chunk = re.findall(PAT,sub_chunk)       

            for pre_token in sub_chunk:
                pre_token = pre_token.encode("utf-8")
                pre_token = list(pre_token)
                tokens = list()
                for token in pre_token:
                    tokens.append(bytes([token]))
                pre_tokens.append(tokens)

        for token in pre_tokens:
            token = tuple(token)
            while True :
                best_pair = None
                best_rank = float('inf')
                
                for index in range(len(token) - 1):

                    pair = (token[index], token[index + 1])
                    if pair in self.merge_rank :
                        rank = self.merge_rank[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair        
                if best_pair is None:
                    break
                else:
                    token = train_tokenizer.merge_token_pair(token, best_pair)

            for byte in token:
                tokenID = self.reversed_vocab[byte]
                result.append(tokenID)

        return result
            
    def encode(self, text: str) -> list[int]:

        if not self.special_tokens:
            return self._encode_ordinary(text)

        results = list()

        pattern = "|".join(re.escape(special_token) for special_token in sorted(self.special_tokens, key=len, reverse=True))

        special_pattern = re.compile(pattern)

        position = 0
        for match in special_pattern.finditer(text):
            special_parts = match.group()
            starts = match.start()
            end = match.end()

            result = self._encode_ordinary(text[position: starts])
            results.extend(result)
            results.append(self.reversed_vocab[special_parts.encode("utf-8")])
            position = end
        results.extend(self._encode_ordinary(text[position: ]))

        return results

    def encode_iterable(self, texts: Iterable[str]) ->Iterator[int]:
        # encode 可迭代对象
        for text in texts:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        vocab = dict()
        merges = list()

        for tokenID_str in raw_vocab:
            tokenID = int(tokenID_str)
            vocab[tokenID] = bytes(raw_vocab[tokenID_str])

        for raw_merge in raw_merges:
            temp_list = list()
            for byte_values in raw_merge:
                token_byte = bytes(byte_values)
                temp_list.append(token_byte)
            merge = tuple(temp_list)
            merges.append(merge)

        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str, merges_filepath: str):
        # save the file
        vocab_data = dict()
        merges_data = list()

        for tokenID in self.vocab:
            vocab_data[str(tokenID)] = list(self.vocab[tokenID])

        for merge in self.merges:
            byte_value_list = list()
            for token_byte in merge:
                byte_value_list.append(list(token_byte))
            merges_data.append(byte_value_list)

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump(merges_data, f)

                    


        
        
        
        
import os
import regex as re

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import BinaryIO

MAX_WORKERS = 4
END_TEXT_TOKEN = '<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(text: str, special_tokens:list[str]) -> list[bytes]:
    # split by special tokens first before pretokenizing
    pattern = "("+"|".join(re.escape(tok) for tok in special_tokens)+")"
    parts = re.split(pattern, text)

    tokens_list = []
    for part in parts:
        if part not in special_tokens:
            matches = re.finditer(PAT, part)
            part_tokens = [s.group().encode('utf-8') for s in matches]
            tokens_list.append(part_tokens)
    return tokens_list


def merge():


def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str])->tuple[dict[int,bytes], list[tuple[bytes,bytes]]]:
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    merges = []

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, END_TEXT_TOKEN.encode("utf-8"))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # Parallelizing pretokenization
    # Use Thread for IO bound task, use ProcessPoolExecutor for CPU-bound task (math)
    with ThreadPoolExecutor(max_workes=MAX_WORKERS) as executor:
        futures = [executor.submit(pretokenize, chunk,  special_tokens) for chunk in chunk_list]
        for future in as_completed(futures):
            pretokens = future.result()
            # merge



    pretokens = [token for tokens in pretokens_list for token in tokens]

    # Merging
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair

    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)

    for i in range(num_merges):
        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]

        index1, index2 = max_pair

        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index)

    return (vocab, merges)

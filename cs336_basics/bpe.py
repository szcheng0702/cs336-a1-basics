import os
import regex as re

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import BinaryIO

MAX_WORKERS = 8
END_TEXT_TOKEN = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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


def pretokenize(text: str, special_tokens: list[str]) -> list[bytes]:
    # split by special tokens first before pretokenizing
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    parts = re.split(pattern, text)

    tokens_list = []
    for part in parts:
        if part in special_tokens:
            # skip special tokens
            continue
        matches = re.finditer(PAT, part)
        part_tokens = [s.group().encode("utf-8") for s in matches]
        tokens_list.append(part_tokens)
    return tokens_list


def merge(counts, index_dict, all_pretokens, max_pair, new_index) -> None:
    affected_pretoken_indices = index_dict[max_pair]
    for idx in affected_pretoken_indices:
        pretoken = all_pretokens[idx]
        new_pretoken, pos_list = [], []
        n = len(pretoken)
        # update
        j = 0
        while j < n:
            if j < n - 1 and (pretoken[j], pretoken[j + 1]) == max_pair:
                new_pretoken.append(new_index)
                pos_list.append(len(new_pretoken) - 1)
                j += 1
            else:
                new_pretoken.append(pretoken[j])
                j += 1

        # update affected pairs
        for pos in pos_list:
            # updated the pair for [prev_token, max_pair[0]]
            if pos > 0:
                # delete old
                if new_pretoken[pos - 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                    if counts[(max_pair[1], max_pair[0])] == 0:
                        index_dict.pop((max_pair[1], max_pair[0]))
                else:
                    counts[(new_pretoken[pos - 1], max_pair[0])] -= 1
                    if counts[(new_pretoken[pos - 1], max_pair[0])] == 0:
                        index_dict.pop((new_pretoken[pos - 1], max_pair[0]))
                # add new
                counts[(new_pretoken[pos - 1], new_index)] += 1
                index_dict[(new_pretoken[pos - 1], new_index)].add(idx)
            # updated the pair for [max_pair[1], post_token]
            if pos < len(new_pretoken) - 1:
                # delete old
                if new_pretoken[pos + 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                    if counts[(max_pair[1], max_pair[0])] == 0:
                        index_dict.pop((max_pair[1], max_pair[0]))
                else:
                    counts[(max_pair[1], new_pretoken[pos + 1])] -= 1
                    if counts[(max_pair[1], new_pretoken[pos + 1])] == 0:
                        index_dict.pop((max_pair[1], new_pretoken[pos + 1]))
                # add new
                counts[(new_index, new_pretoken[pos + 1])] += 1
                index_dict[(new_index, new_pretoken[pos + 1])].add(idx)
        all_pretokens[idx] = new_pretoken


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = MAX_WORKERS,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {}
    vocab = {x: bytes([x]) for x in range(0, 256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    merges = []

    # Read file
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, END_TEXT_TOKEN.encode("utf-8")
        )

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # Parallelizing pretokenization
    # Use Thread for IO bound task, use ProcessPoolExecutor for CPU-bound task (math)
    all_pretokens = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(pretokenize, chunk, special_tokens) for chunk in chunk_list
        ]
        for future in as_completed(futures):
            pretokens = future.result()
            all_pretokens.append(pretokens)

    # Merging
    import pdb

    pdb.set_trace()
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair

    for j, pretoken in enumerate(all_pretokens):
        for i in range(len(pretoken) - 1):
            p = (pretoken[i], pretoken[i + 1])
            counts[p] += 1
            index_dict[p].add(j)

    for i in range(num_merges):
        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],
                x[0],
            ),
        )[0]

        index1, index2 = max_pair
        new_index = 256 + len(special_tokens) + i
        import pdb

        pdb.set_trace()
        vocab[new_index] = vocab[index1] + vocab[index2]
        merge(counts, index_dict, all_pretokens, max_pair, new_index)
        merges.append((vocab[index1], vocab[index2]))

    return vocab, merges

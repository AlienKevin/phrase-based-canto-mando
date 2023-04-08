common_trad_chars = None

with open("common_trad_chars.txt", "r") as input_file:
    common_trad_chars = set(input_file.read())

print("A sample of common traditional characters: ", list(common_trad_chars)[0:10])

from collections import defaultdict

can2man_table = defaultdict(list)

with open("can2man_phrase_table.txt", "r") as input_file:
  for line in input_file.read().splitlines():
    [can, man] = line.split("|")
    can2man_table[can].append(man)

def longest_match_translate(s, phrase_table):
    man_phrases: list[list[str]] = []
    oov_word = ""
    while s:
        longest_match = None
        for phrase in phrase_table:
            if s.startswith(phrase) and (longest_match is None or len(phrase) > len(longest_match)):
                longest_match = phrase
        if longest_match:
            if len(oov_word) > 0:
                man_phrases.append([oov_word])
                oov_word = ""
            can_original = [longest_match] if len(longest_match) <= 1 and all(c in common_trad_chars for c in longest_match) else []
            man_phrase = phrase_table[longest_match]
            man_phrases.append(can_original + man_phrase)
            s = s[len(longest_match):].lstrip()
        else:
            oov_word += s[0]
            s = s[1:].lstrip()
    if len(oov_word) > 0:
        man_phrases.append([oov_word])
    # Merge anchor phrases (those with a single mandarin translation)
    i = 0
    merged_man_phrases = []
    while i < len(man_phrases):
        merged_phrase = ""
        while i < len(man_phrases) and len(man_phrases[i]) == 1:
            merged_phrase += man_phrases[i][0]
            i += 1
        if len(merged_phrase) > 0:
            merged_man_phrases.append([merged_phrase])
            merged_phrase = ""
        else:
            merged_man_phrases.append(man_phrases[i])
            i += 1
    return merged_man_phrases

from transformers import BertTokenizerFast, GPT2LMHeadModel
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = GPT2LMHeadModel.from_pretrained('ckiplab/gpt2-base-chinese')

import torch

# https://huggingface.co/docs/transformers/perplexity
def get_most_fluent_sentence_index(candidates: list[str]) -> int:
    encodings = [tokenizer(candidate, return_tensors="pt") for candidate in candidates]
    ppls = []
    for encoding in encodings:
        target_ids_list = []
        seq_len = encoding.input_ids.size(1)
        for end_loc in range(2, seq_len + 1, 2):
            target_ids = encoding.input_ids[0].clone()
            target_ids[end_loc:] = -100
            target_ids_list.append(target_ids)
        target_ids = torch.stack(target_ids_list)
        input_ids = encoding.input_ids.expand(target_ids.shape)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = outputs.loss.item() * seq_len
            ppls.append(ppl)
    return torch.argmin(torch.tensor(ppls))

import regex

# match any unicode punctuation character and anything after it
punctuation_pattern = regex.compile(r"\p{P}+.*", flags=regex.UNICODE)
chinese_char_pattern = regex.compile(r"[\u4e00-\u9fff]")

def chop_off_at_punctuation(s: str) -> str:
    match = punctuation_pattern.search(s)
    if match:
        index = match.start()
        return s[:index]
    else:
        return s

def chop_off_at_canto_char(s: str) -> str:
    for i, c in enumerate(s):
        if chinese_char_pattern.match(c) and not c in common_trad_chars:
            return s[:i]
    return s

def flatten(l):
    return [item for sublist in l for item in sublist]

def can2man(s: str) -> str:
    s = s.replace(" ", "")
    man_phrases = longest_match_translate(s, can2man_table)
    # print(man_phrases)
    for i, phrases in enumerate(man_phrases):
        if len(phrases) == 1:
            continue
        else:
            j = i + 1
            while j < len(man_phrases) and man_phrases[j] == 1:
                j += 1
            backward_context = "".join(flatten(man_phrases[max(0, i - 10):i]))
            forward_context = "".join(flatten(man_phrases[i + 1:j]))
            # forward context is too small
            while len(forward_context) < 10 and j < len(man_phrases):
                forward_context += man_phrases[j][0]
                j += 1
            forward_context = chop_off_at_canto_char(chop_off_at_punctuation(forward_context))
            # print(f"i={i} backward_context={backward_context}, forward_context={forward_context}")
            candidates = [backward_context + phrase + forward_context for phrase in man_phrases[i]]
            j = get_most_fluent_sentence_index(candidates)
            man_phrases[i] = [man_phrases[i][j]]
    # print(man_phrases)
    return "".join(flatten(man_phrases))

assert can2man("唔該你細聲啲，我喺度做緊嘢。") == "請你小聲點，我在這裡正在做東西。"

from tqdm import tqdm
import sys

i = sys.argv[1]
with open(f"can/can-{i}.txt", "r") as input_file, open(f"man/man-{i}.txt", "w+") as output_file:
    print(f"Translating can-{i}.txt and saving results to man-{i}.txt")
    for line in tqdm(input_file.read().splitlines()):
        output_file.write(can2man(line) + "\n")

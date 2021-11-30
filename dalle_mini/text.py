"""
Utilities for processing text.
"""

import html
import math
import random
import re
from pathlib import Path

import ftfy
from huggingface_hub import hf_hub_download
from unidecode import unidecode

# based on wiki word occurence
person_token = [("a person", 282265), ("someone", 121194), ("somebody", 12219)]
temp_token = "xtokx"  # avoid repeating chars


class HashtagProcessor:
    # Adapted from wordninja library
    # We use our wikipedia word count + a good heuristic to make it work
    def __init__(self):
        wiki_word_frequency = hf_hub_download(
            "dalle-mini/dalle-mini", filename="enwiki-words-frequency.txt"
        )
        self._word_cost = (
            l.split()[0] for l in Path(wiki_word_frequency).read_text().splitlines()
        )
        self._word_cost = {
            str(k): math.log(float(i + 1)) for i, k in enumerate(self._word_cost)
        }
        self._max_word = max(len(x) for x in self._word_cost.keys())
        self._SPLIT_RE = re.compile("[^a-zA-Z0-9']+")

    def __call__(self, s):
        """Uses dynamic programming to infer the location of spaces in a string without spaces."""
        l = [self._split(x) for x in self._SPLIT_RE.split(s)]
        return " ".join([item for sublist in l for item in sublist])

    def _split(self, s):
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self._max_word) : i]))
            return min(
                (c + self._word_cost.get(s[i - k - 1 : i].lower(), 9e999), k + 1)
                for k, c in candidates
            )

        # Build the cost array
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            newToken = True
            if not s[i - k : i] == "'":  # ignore a lone apostrophe
                if len(out) > 0:
                    # re-attach split 's and split digits
                    if out[-1] == "'s" or (
                        s[i - 1].isdigit() and out[-1][0].isdigit()
                    ):  # digit followed by digit
                        out[-1] = (
                            s[i - k : i] + out[-1]
                        )  # combine current token with previous token
                        newToken = False

            if newToken:
                out.append(s[i - k : i])

            i -= k

        return reversed(out)


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace(
            "<person>", f" {random.choices(*tuple(zip(*person_token)))[0]} ", 1
        )
    return t


def fix_html(t):
    # from OpenAI CLIP
    return html.unescape(html.unescape(t))


def replace_punctuation_with_commas(t):
    return re.sub("[()[\].,|:;?!=+~\-\/{}]", ",", t)


def simplify_quotes(t):
    return re.sub("""['"`]""", ' " ', t)


def merge_quotes(t):
    return re.sub('(\s*"+\s*)+', ' " ', t)


def remove_comma_numbers(t):
    def _f(t):
        return re.sub("(\d),(\d{3})", r"\1\2", t)

    return _f(_f(t))


def pre_process_dot_numbers(t):
    return re.sub("(\w)\.(\w)", fr"\1{temp_token}dot{temp_token}\2", t)


def post_process_dot_numbers(t):
    return re.sub(f"{temp_token}dot{temp_token}", ".", t)


def pre_process_quotes(t):
    # allows quotes only for 's, 't, 'd, 'm, 'll, 're, 've
    return re.sub(
        r"'(?=([stdm]|(ll)|(re)|(ve)|(ll))\b)", fr"{temp_token}quote{temp_token}", t
    )


def post_process_quotes(t):
    return re.sub(f"{temp_token}quote{temp_token}", "'", t)


def pre_process_dates(t):
    return re.sub("(\d)/(\d)", fr"\1{temp_token}slash{temp_token}\2", t)


def post_process_dates(t):
    return re.sub(f"{temp_token}slash{temp_token}", "/", t)


def merge_commas(t):
    return re.sub("(\s*,+\s*)+", ", ", t)


def add_space_after_commas(t):
    return re.sub(",", ", ", t)


def handle_special_chars(t):
    "Handle special characters"
    # replace "-" with a space when between words without space
    t = re.sub("(\w)-(\w)", r"\1 \2", t)
    # always add space around some characters
    return re.sub("([%&\/$*])", r" \1 ", t)


def expand_hashtags(t, hashtag_processor):
    "Remove # and try to split words"
    return re.sub("#(\w+)", lambda m: hashtag_processor(m.group(1)), t)


_re_ignore_chars = r"[_#\\]"


def ignore_chars(t):
    "Ignore useless characters"
    return re.sub(_re_ignore_chars, " ", t)


def remove_extra_spaces(t):
    "Remove extra spaces (including \t and \n)"
    return re.sub("\s+", " ", t)


def remove_repeating_chars(t):
    "If the same character is present 4+ times (not 3 because of roman 'VIII'), replace with single instance"
    return re.sub(r"(\D)(\1{3,})", r"\1", t)


def remove_urls(t):
    return re.sub(r"http\S+", "", t)


def remove_html_tags(t):
    return re.sub("<[^<]+?>", "", t)


def remove_first_last_commas(t):
    t = t.strip()
    t = t[:-1] if t and t[-1] == "," else t
    t = t[1:] if t and t[0] == "," else t
    return t.strip()


def remove_wiki_ref(t):
    t = re.sub(r"\A\s*\[\d+\]", "", t)
    return re.sub(r"\[\d+\]\s*\Z", "", t)


class TextNormalizer:
    "Normalize text"

    def __init__(self):
        self._hashtag_processor = HashtagProcessor()

    def __call__(self, t):
        # fix some characters
        t = ftfy.fix_text(t)
        # fix html
        t = fix_html(t)
        # decode and simplify text: see unidecode library
        t = unidecode(t)
        # lower case
        t = t.lower()
        # replace <PERSON> (for CC12M)
        t = replace_person_token(t)
        # remove wiki reference (for WIT)
        t = remove_wiki_ref(t)
        # remove html tags
        t = remove_html_tags(t)
        # remove urls
        t = remove_urls(t)
        # remove commas in numbers
        t = remove_comma_numbers(t)
        # handle dots in numbers and quotes - Part 1
        t = pre_process_dot_numbers(t)
        t = pre_process_quotes(t)
        t = pre_process_dates(t)
        # handle special characters
        t = handle_special_chars(t)
        # handle hashtags
        t = expand_hashtags(t, self._hashtag_processor)
        # ignore useless characters
        t = ignore_chars(t)
        # simplify quotes
        t = simplify_quotes(t)
        # all punctuation becomes commas
        t = replace_punctuation_with_commas(t)
        # handle dots in numbers and quotes - Part 2
        t = post_process_dot_numbers(t)
        t = post_process_quotes(t)
        t = post_process_dates(t)
        # handle repeating characters
        t = remove_repeating_chars(t)
        # merge quotes
        t = merge_quotes(t)
        # merge commas
        t = merge_commas(t)
        # remove multiple spaces
        t = remove_extra_spaces(t)
        # remove first and last comma
        t = remove_first_last_commas(t)
        # always start with a space
        return f" {t}"

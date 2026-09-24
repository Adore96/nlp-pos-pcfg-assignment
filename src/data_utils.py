import csv
import random
import re

# Curly quotes / dashes as used on dailymirror.lk -> the ASCII forms the
# Treebank tokenizer understands (it then turns " into `` and '').
_UNICODE_PUNCT = {
    '‘': ' ` ', '’': "'", '“': '"', '”': '"',
    '–': ' -- ', '—': ' -- ', ' ': ' ',
}
# The Penn Treebank writes brackets as tokens, not as the characters.
_BRACKETS = {'(': '-LRB-', ')': '-RRB-', '[': '-LRB-', ']': '-RRB-',
             '{': '-LCB-', '}': '-RCB-'}


def clean_sent(sent):
    """Remove Penn Treebank empty-category traces (the -NONE- tag)."""
    return [(w, t) for w, t in sent if t != '-NONE-']


def load_clean_treebank():
    from nltk.corpus import treebank
    return [clean_sent(s) for s in treebank.tagged_sents()]


def train_test_indices(n, seed=42, train_frac=0.9):
    """The 90/10 split used for the tagger in train_tagger.ipynb, as sentence
    indices, so the PCFG can be trained on exactly the same sentences.
    random.shuffle's permutation depends only on the seed and list length,
    so shuffling range(n) reproduces the tagger's shuffle of the sentences."""
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    split = int(n * train_frac)
    return idx[:split], idx[split:]


def ptb_tokenize(sentence):
    """Tokenize a raw sentence the way the Penn Treebank does ($ 97.36,
    it 's, did n't, `` quotes '', -LRB- brackets -RRB-)."""
    from nltk.tokenize import TreebankWordTokenizer
    for ch, repl in _UNICODE_PUNCT.items():
        sentence = sentence.replace(ch, repl)
    tokens = [_BRACKETS.get(t, t) for t in TreebankWordTokenizer().tokenize(sentence)]
    # The tokenizer splits the final '.' off abbreviations such as "a.m.";
    # the Treebank keeps the abbreviation whole and adds a separate '.'.
    if len(tokens) >= 2 and tokens[-1] == '.' and re.fullmatch(r'(?:[A-Za-z]\.)+[A-Za-z]', tokens[-2]):
        tokens[-2] += '.'
    return tokens


def load_dm_sentences(path='../data/raw/daily_mirror_sentences.csv'):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_gold_tagged(path='../data/tagged/daily_mirror_tagged.tsv'):
    """Reads the hand-tagged Daily Mirror sentences: a TSV with columns
    sent_id, token_id, word, tag. Returns {sent_id: [(word, tag), ...]}."""
    sents = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            sents.setdefault(int(row['sent_id']), []).append((row['word'], row['tag']))
    return sents

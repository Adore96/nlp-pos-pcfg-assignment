def clean_sent(sent):
    """Remove Penn Treebank empty-category traces (the -NONE- tag)."""
    return [(w, t) for w, t in sent if t != '-NONE-']


def load_clean_treebank():
    from nltk.corpus import treebank
    return [clean_sent(s) for s in treebank.tagged_sents()]
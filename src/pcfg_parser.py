"""PCFG induced from the NLTK Penn Treebank sample, plus a Viterbi CKY parser.

Grammar pipeline (induce_grammar):
  1. Clean each gold tree: drop -NONE- traces and constituents left empty,
     strip function tags / co-indices (NP-SBJ-1 -> NP), wrap in a TOP node.
  2. Binarize (Chomsky normal form, horizontal Markovization) and collapse
     unary chains between phrasal nodes, so every rule is A -> B C, A -> B
     (phrase -> POS tag or TOP -> phrase) or TAG -> word.
  3. Count rules and take relative frequencies (maximum likelihood).

Unknown words: every training word seen only once also counts as an example
of its "signature" class (UNK-CAP-s, UNK-NUM, UNK-ing, ...), so the lexicon
has rules like NNP -> UNK-CAP. At parse time a word never seen in training is
replaced by its signature. This lets the parser tag words it has never seen,
which the plain HMM tagger cannot do.
"""
import json
import math
import re
from collections import Counter, defaultdict

from nltk import Nonterminal, Production, Tree, induce_pcfg
from nltk.grammar import PCFG, ProbabilisticProduction

TOP = 'TOP'
PUNCT_TAGS = {',', '.', ':', '``', "''", '-LRB-', '-RRB-', '#', '$'}
# Tags evalb ignores when scoring brackets.
EVALB_IGNORE_TAGS = {',', '.', ':', '``', "''"}
LOG_FLOOR = -30.0   # log-prob for a word/tag pair the lexicon has never seen


# --------------------------------------------------------------- trees ----

def _strip_label(label):
    if label.startswith('-'):          # -LRB-, -NONE- ... are real labels
        return label
    return re.split(r'[-=]', label)[0] or label


def clean_tree(tree):
    """Remove traces and function tags; returns None if nothing is left."""
    if isinstance(tree, str):
        return tree
    if tree.label() == '-NONE-':
        return None
    kids = [k for k in (clean_tree(c) for c in tree) if k is not None]
    if not kids:
        return None
    return Tree(_strip_label(tree.label()), kids)


def prepare_tree(tree, horz_markov=1, vert_markov=0):
    """Clean + binarize a gold treebank tree for grammar induction."""
    t = clean_tree(tree)
    if t.label() != TOP:
        t = Tree(TOP, [t])
    t.chomsky_normal_form(horzMarkov=horz_markov, vertMarkov=vert_markov)
    t.collapse_unary(collapsePOS=False, collapseRoot=False)
    return t


def debinarize(tree):
    """Undo prepare_tree's binarization so trees compare with gold ones."""
    t = tree.copy(deep=True)
    t.un_chomsky_normal_form()
    return t


# ------------------------------------------------------ unknown words ----

def word_signature(word):
    """Coarse class for a word the lexicon hasn't seen (Berkeley-parser
    style): capitalisation, digits, hyphens and a telling suffix."""
    sig = 'UNK'
    if any(c.isdigit() for c in word):
        sig += '-NUM'
        if re.fullmatch(r'[\d.,:/-]+', word):
            return sig                      # a pure number/date/time
    if word[:1].isupper():
        sig += '-CAP'
        if word.isupper():
            sig += '-ALL'
    elif any(c.isupper() for c in word):
        sig += '-MIXED'
    if '-' in word:
        sig += '-HYPH'
    lower = word.lower()
    for suffix in ('ing', 'ed', 'ly', 'ion', 'er', 'est', 'al', 'ive', 'ous',
                   'able', 'ity', 'ment', 'ness', 'ist', 'ism', 'ic', 's'):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
            return sig + '-' + suffix
    return sig


# ------------------------------------------------------------ grammar ----

def induce_grammar(gold_trees, horz_markov=1, vert_markov=0):
    """Returns an nltk PCFG with start symbol TOP."""
    trees = [prepare_tree(t, horz_markov, vert_markov) for t in gold_trees]
    word_freq = Counter(w for t in trees for w in t.leaves())
    productions = []
    for t in trees:
        for p in t.productions():
            productions.append(p)
            if p.is_lexical() and word_freq[p.rhs()[0]] == 1:
                productions.append(Production(p.lhs(), [word_signature(p.rhs()[0])]))
    return induce_pcfg(Nonterminal(TOP), productions)


def save_grammar(grammar, path, header=()):
    """One rule per line: `A -> B C [p]`, terminals JSON-quoted (`NN -> "dog" [p]`).
    nltk's own PCFG.fromstring can't read binarized labels like S|<VP>,
    and str(grammar) rounds probabilities, hence this simple format."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(';; PCFG induced from the NLTK Penn Treebank sample (train split)\n')
        for line in header:
            f.write(f';; {line}\n')
        f.write(f';; start symbol: {grammar.start()}   rules: {len(grammar.productions())}\n')
        for p in grammar.productions():
            rhs = ' '.join(json.dumps(x) if isinstance(x, str) else str(x) for x in p.rhs())
            f.write(f'{p.lhs()} -> {rhs} [{p.prob()!r}]\n')


def load_grammar(path):
    productions = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith(';;') or not line.strip():   # ('#' is a POS tag)
                continue
            rule, prob = line.rstrip('\n').rsplit(' [', 1)
            lhs, rhs = rule.split(' -> ', 1)
            if rhs.startswith('"'):
                rhs = [json.loads(rhs)]
            else:
                rhs = [Nonterminal(x) for x in rhs.split(' ')]
            productions.append(ProbabilisticProduction(Nonterminal(lhs), rhs, prob=float(prob[:-1])))
    return PCFG(Nonterminal(TOP), productions)


# ------------------------------------------------------------- parser ----

class ViterbiCKYParser:
    """Most-probable-parse CKY for the binarized PCFG.

    nltk's own ViterbiParser is far too slow for a treebank-sized grammar, so
    this indexes binary rules by their (left, right) child pair, handles the
    remaining unary rules with a closure step per cell, and keeps only the
    `beam` best labels per cell.
    """

    def __init__(self, grammar, beam=40):
        self.grammar = grammar
        self.beam = beam
        self.start = str(grammar.start())
        self.binary = defaultdict(list)        # (B, C) -> [(A, logp)]
        self.unary = defaultdict(list)         # B -> [(A, logp)]
        self.lexical = defaultdict(dict)       # word -> {TAG: logp}
        for p in grammar.productions():
            lp = math.log(p.prob())
            a, rhs = str(p.lhs()), p.rhs()
            if p.is_lexical():
                self.lexical[rhs[0]][a] = lp
            elif len(rhs) == 2:
                self.binary[(str(rhs[0]), str(rhs[1]))].append((a, lp))
            else:
                self.unary[str(rhs[0])].append((a, lp))
        self.tags = {t for d in self.lexical.values() for t in d}

    def known(self, word):
        return word in self.lexical

    def _lexical_scores(self, word, tag=None):
        entry = self.lexical.get(word) or self.lexical.get(word_signature(word))
        if entry is None:                      # signature never seen in training
            entry = self.lexical.get('UNK', {})
        if tag is not None:                    # tag supplied by a POS tagger
            return {tag: entry.get(tag, LOG_FLOOR)}
        return dict(entry)

    def _unary_closure(self, cell):
        agenda = list(cell)
        while agenda:
            b = agenda.pop()
            b_lp = cell[b][0]
            for a, lp in self.unary.get(b, ()):
                score = b_lp + lp
                if a not in cell or score > cell[a][0]:
                    cell[a] = (score, ('unary', b))
                    agenda.append(a)

    def _prune(self, cell):
        if len(cell) > self.beam:
            keep = sorted(cell.items(), key=lambda kv: kv[1][0], reverse=True)[:self.beam]
            # the unary backpointers of kept labels may point at dropped ones,
            # so keep those too
            kept = dict(keep)
            for label, (_, bp) in keep:
                while bp[0] == 'unary' and bp[1] not in kept:
                    kept[bp[1]] = cell[bp[1]]
                    bp = cell[bp[1]][1]
            cell.clear()
            cell.update(kept)

    def parse(self, words, tags=None):
        """Returns (tree, logprob), or (None, None) if no full parse exists.
        With `tags` the preterminals are fixed to the given POS tags."""
        n = len(words)
        chart = [[None] * (n + 1) for _ in range(n + 1)]
        for i, w in enumerate(words):
            cell = {t: (lp, ('word', w)) for t, lp in
                    self._lexical_scores(w, tags[i] if tags else None).items()}
            self._unary_closure(cell)
            self._prune(cell)
            chart[i][i + 1] = cell
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell = {}
                for k in range(i + 1, j):
                    left, right = chart[i][k], chart[k][j]
                    if not left or not right:
                        continue
                    for b, (b_lp, _) in left.items():
                        for c, (c_lp, _) in right.items():
                            rules = self.binary.get((b, c))
                            if not rules:
                                continue
                            base = b_lp + c_lp
                            for a, lp in rules:
                                score = base + lp
                                if a not in cell or score > cell[a][0]:
                                    cell[a] = (score, ('binary', k, b, c))
                self._unary_closure(cell)
                self._prune(cell)
                chart[i][j] = cell
        top = chart[0][n]
        if not top or self.start not in top:
            return None, None
        return self._build(chart, 0, n, self.start), top[self.start][0]

    def _build(self, chart, i, j, label):
        _, bp = chart[i][j][label]
        if bp[0] == 'word':
            return Tree(label, [bp[1]])
        if bp[0] == 'unary':
            return Tree(label, [self._build(chart, i, j, bp[1])])
        _, k, b, c = bp
        return Tree(label, [self._build(chart, i, k, b), self._build(chart, k, j, c)])


# ----------------------------------------------------------- PARSEVAL ----

def brackets(tree):
    """Labeled brackets (label, start, end) of a clean (debinarized) tree,
    evalb-style: preterminals and the TOP node are not brackets, and
    punctuation tokens don't count towards span positions."""
    out = Counter()
    pos = [0]

    def walk(t):
        if isinstance(t[0], str):                 # preterminal
            if t.label() not in EVALB_IGNORE_TAGS:
                pos[0] += 1
            return
        start = pos[0]
        for child in t:
            walk(child)
        if t.label() != TOP and pos[0] > start:
            out[(t.label(), start, pos[0])] += 1

    walk(tree)
    return out


def parseval(gold_tree, pred_tree):
    """Returns (matched, n_gold, n_pred) labeled-bracket counts."""
    g, p = brackets(gold_tree), brackets(pred_tree)
    return sum((g & p).values()), sum(g.values()), sum(p.values())


def tree_pos(tree):
    return [t for _, t in tree.pos()]

# POS Tagging and PCFG Parsing on Sri Lankan News

MSc Natural Language Processing coursework. The project trains a Hidden Markov Model (HMM) part-of-speech tagger and a probabilistic context-free grammar (PCFG) parser on the NLTK Penn Treebank sample. It then evaluates both on 150 hand-tagged sentences from the Sri Lankan newspaper *Daily Mirror* ([dailymirror.lk](https://www.dailymirror.lk)).

The goal is to measure how well models trained on 1980s Wall Street Journal text transfer to present-day Sri Lankan English news.

## Results at a glance

**Daily Mirror test set** (150 sentences, 2,011 tokens):

| System | Token accuracy | Weighted F1 | Macro F1 | Sentence exact match |
|---|---|---|---|---|
| HMM tagger | 82.7% | 0.830 | 0.733 | 13.3% |
| PCFG (joint parse + tag) | **93.0%** | **0.930** | **0.873** | **42.0%** |

- HMM accuracy is 93.9% on known words but only 25.7% on the 331 unknown (out-of-vocabulary) tokens.
- The joint PCFG parses 100% of the sentences. The PCFG fed HMM tags parses 96%.
- Most frequent HMM confusions: `NN`/`NNS`, `VBD`/`VBN`, `NNP`/`PRP`, `JJ`/`NN`.

**Treebank held-out split** (10%, 392 sentences):

- HMM tagger accuracy: 91.5%.
- PCFG labeled PARSEVAL F1 (sentences up to 40 tokens, beam 40):

| Tag source | Labeled P | Labeled R | Labeled F1 | Coverage |
|---|---|---|---|---|
| Gold tags | 0.819 | 0.751 | 0.784 | 98.9% |
| Joint (parser picks tags) | 0.785 | 0.726 | 0.755 | 99.2% |
| HMM tags (pipeline) | 0.673 | 0.614 | 0.642 | 98.1% |

Full numbers are in [`results/`](results/).

## Repository layout

```
data/
  raw/daily_mirror_sentences.csv     150 collected sentences with section and source URL
  tagged/daily_mirror_tagged.tsv     gold POS tags (sent_id, token_id, word, tag)
models/
  hmm_tagger.pkl                     trained HMM tagger (pickled)
  pcfg_grammar.txt                   induced PCFG, custom text format
notebooks/
  explore_treebank.ipynb             corpus exploration
  train_tagger.ipynb                 HMM training and held-out accuracy
  build_pcfg.ipynb                   grammar induction, ablation, held-out PARSEVAL
  evaluate.ipynb                     evaluation on the Daily Mirror set
results/                             metrics, predictions, parse trees, figures
scraper/                             sentence collection scripts
src/
  data_utils.py                      treebank cleaning, train/test split, PTB tokenizer, loaders
  tagger.py                          Lidstone estimator used by the HMM
  pcfg_parser.py                     tree preparation, grammar induction, CKY parser, PARSEVAL
  evaluate.py                        accuracy, per-tag scores, confusion matrices
```

## Method

### Data preparation

`src/data_utils.py` loads the NLTK Penn Treebank sample (3,914 sentences) and removes `-NONE-` empty-category traces, because real news text never contains them. Sentences are shuffled with seed 42 and split 90/10 (3,522 train, 392 test). The tagger and the PCFG use the same split.

### HMM tagger

The tagger is a supervised bigram HMM (`nltk.tag.hmm.HiddenMarkovModelTrainer`). Emission and transition probabilities use Lidstone smoothing with gamma = 0.1.

### PCFG parser

The grammar is induced from the training trees:

1. Clean each tree: remove traces and function tags, and add a `TOP` root.
2. Binarize with horizontal Markovization 1 and parent annotation (vertical Markovization 1).
3. Collapse unary chains.
4. Map rare and unseen words to word-shape signature classes (`UNK` handling). Training hapaxes count toward these classes.

The result has 18,628 rules. An ablation over horizontal (1, 2) and vertical (0, 1) Markovization settings picked this configuration by held-out F1.

`ViterbiCKYParser` is a custom beam-pruned CKY parser, because NLTK's `ViterbiParser` is too slow for this grammar. It runs in two modes:

- **Joint**: the parser chooses the POS tags while parsing.
- **Fixed tags**: the parser takes tags from another source (gold or HMM).

### Test set

Sentences come from Daily Mirror articles across these sections: international, business, opinion, news features and breaking news. Filters keep declarative sentences of 6–15 words. The sentences are tokenized with `ptb_tokenize()` and tagged by hand using the Penn Treebank tagset.

## Setup

Requires Python 3 on Windows (paths below use the Windows virtualenv layout).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -c "import nltk; nltk.download('treebank')"
```

## Reproducing the results

Run the notebooks from `notebooks/`, in this order. Each notebook adds the repository root to `sys.path` and uses `../` relative paths, so the working directory must be `notebooks/`.

1. `train_tagger.ipynb` writes `models/hmm_tagger.pkl`.
2. `build_pcfg.ipynb` writes `models/pcfg_grammar.txt` and `results/treebank_parseval.json`. The ablation takes several minutes.
3. `evaluate.ipynb` writes the tagging and parsing results to `results/`.

To run a notebook headlessly:

```bash
cd notebooks && ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace train_tagger.ipynb
```

### Notes

- `hmm_tagger.pkl` references `src.tagger.lidstone_estimator` by module path. Do not rename or move that function, and make sure `src` is importable before you unpickle the tagger.
- Load the grammar with `src.pcfg_parser.load_grammar()`, not `nltk.PCFG.fromstring`. The file uses its own format: `LHS -> RHS [p]`, JSON-quoted terminals, and `;;` comments (`#` is a POS tag).
- Gold tokens in `daily_mirror_tagged.tsv` must match `ptb_tokenize()` output exactly. If you change the tokenizer, re-check the gold file.

## Output files

| File | Contents |
|---|---|
| `results/tagging_metrics.json` | Accuracy, precision, recall, F1, known/unknown split, top confusions, parse coverage |
| `results/tagging_predictions.tsv` | Per-token gold, HMM and PCFG tags |
| `results/parse_results.csv` | Per-sentence accuracy, OOV counts, log-probabilities, joint parse tree |
| `results/parse_trees.txt` | Readable parse trees for all 150 sentences |
| `results/treebank_parseval.json` | Markovization ablation and held-out PARSEVAL |
| `results/confusion_matrix*.csv` | Tag confusion matrices |
| `results/figures/*.png` | Confusion matrix, per-tag F1, per-sentence accuracy, accuracy vs OOV rate |

## Collecting new sentences

The scripts in `scraper/` run from inside that folder and write `dm_sentences.csv` and `dm_sentences.txt` there.

- `dm_sentence_scraper.py` reads RSS feeds and listing pages with headless Playwright. It drives the installed Chrome or Edge, so `playwright install` is not needed. The site often blocks automated requests with 403 responses.
- `process_manual_articles.py` is the fallback, and the source of the current 150 sentences. It parses article bodies pasted into `manual_articles.txt`, where each article starts with a `URL: ...` line. It picks sentences round-robin across articles, up to 150. It merges into an existing `dm_sentences.csv`, so delete that file to rebuild from scratch.

Copy the final `scraper/dm_sentences.csv` to `data/raw/daily_mirror_sentences.csv`.

## Data sources

- Penn Treebank sample distributed with NLTK (about 10% of the WSJ portion).
- Sentences quoted from *Daily Mirror* (Sri Lanka) articles for academic evaluation only. Source URLs are recorded per sentence.

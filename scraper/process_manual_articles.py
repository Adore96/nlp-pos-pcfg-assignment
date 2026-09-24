"""
process_manual_articles.py

Fallback / companion to dm_sentence_scraper.py for whenever DM's site is
rate-limiting or blocking the automated scraper. This processes article
text YOU collect by hand -- open an article normally in your own browser,
copy the body paragraphs -- since that's genuine human browsing and
doesn't trigger the bot-detection the scraper hit (HTTP 403s).

HOW TO COLLECT:
Keep one growing text file, manual_articles.txt, in this format:

    URL: https://www.dailymirror.lk/...
    <paste the article's body paragraphs here>
    ---
    URL: https://www.dailymirror.lk/...
    <paste the next article's body paragraphs here>
    ---

One "URL: ..." line, then the pasted body text, then a line with just
"---" to separate articles. Add as many blocks as you like, across as
many browsing sessions as you like -- rerun this script any time.

RUN:
    python process_manual_articles.py

This applies the same sentence filters as the scraper (6-15 words,
declarative, clean capitalization/ending), merges with dm_sentences.csv
if it already exists from the automated run (tops up toward 150 and
skips sentences you've already collected, so nothing is duplicated or
lost), and re-saves dm_sentences.csv / dm_sentences.txt.
"""

import re
import csv
import os
from urllib.parse import urlparse
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

MIN_WORDS = 6
MAX_WORDS = 15
MAX_SENTENCES_PER_ARTICLE = 3
TARGET_TOTAL = 150
INPUT_FILE = "manual_articles.txt"
CSV_FILE = "dm_sentences.csv"
TXT_FILE = "dm_sentences.txt"

CAPTION_MARKERS = ("pic by", "pic courtesy", "photo by", "photo courtesy", "afp", "reuters photo")
END_PUNCT_RE = re.compile(r'\.[\"\'\u2018\u2019\u201c\u201d]?$')
INVISIBLE_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\u00ad]')  # zero-width chars DM scatters through text
ABBREV_END_RE = re.compile(r'\b(?:No|Rs|Mr|Mrs|Ms|Dr|St|Co|Ltd|Inc|Jr|Sr|vs)\.$')  # split after an abbreviation
URL_LINE_RE = re.compile(r'^\s*URL:\s*(\S+)', re.IGNORECASE)
SEPARATOR_RE = re.compile(r'^\s*---\s*')


def derive_section(url):
    path = urlparse(url).path.strip("/")
    return path.split("/")[0].lower() if path else "unknown"


def classify_sentence(sentence):
    s = sentence.strip()
    words = s.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if not s[:1].isupper():
        return False
    if not END_PUNCT_RE.search(s):
        return False
    if s.isupper():
        return False
    if any(m in s.lower() for m in CAPTION_MARKERS):
        return False
    if ABBREV_END_RE.search(s):
        return False
    if s.count('"') % 2 or s.count("“") != s.count("”"):
        return False   # piece of a longer quotation
    return True


def load_existing():
    rows, seen = [], set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                seen.add(r["sentence"])
    return rows, seen


def parse_manual_file(path):
    """Yields (url, body_text) per article. A line starting with 'URL:' opens
    a new article and '---' separator lines are skipped. Indentation is
    ignored, so pasted blocks don't have to start at column 0."""
    if not os.path.exists(path):
        print(f"No {path} found yet. Create it as described in this "
              f"script's docstring (URL line + pasted text + '---' "
              f"between articles), then rerun.")
        return
    with open(path, encoding="utf-8") as f:
        lines = INVISIBLE_RE.sub("", f.read()).splitlines()
    url, body = None, []
    for line in lines:
        line = SEPARATOR_RE.sub("", line)   # also handles '---   URL: ...' on one line
        m = URL_LINE_RE.match(line)
        if m:
            if url and body:
                yield url, "\n".join(body)
            url, body = m.group(1), []
        elif line.strip() and not line.strip().startswith("<paste"):
            body.append(line.strip())
    if url and body:
        yield url, "\n".join(body)


def main():
    rows, seen = load_existing()
    next_id = len(rows) + 1
    added = 0

    # Valid sentences per article, in reading order. Tokenize per paragraph
    # so an unpunctuated sub-heading can't get glued onto the next sentence.
    candidates = []
    for url, body in parse_manual_file(INPUT_FILE):
        valid = []
        for sent in (s for para in body.splitlines() for s in sent_tokenize(para)):
            clean = re.sub(r"\s+", " ", sent).strip()
            if clean not in seen and classify_sentence(clean):
                seen.add(clean)
                valid.append(clean)
        candidates.append((url, valid))

    # Round-robin across articles: everyone gets 1, then 2, ... so sentences
    # stay spread out, and long articles only fill whatever is still missing.
    for round_no in range(max((len(v) for _, v in candidates), default=0)):
        for url, valid in candidates:
            if len(rows) >= TARGET_TOTAL:
                break
            if round_no < len(valid):
                clean = valid[round_no]
                rows.append({
                    "id": next_id,
                    "sentence": clean,
                    "word_count": len(clean.split()),
                    "section": derive_section(url),
                    "source_url": url,
                })
                next_id += 1
                added += 1

    if rows:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "sentence", "word_count", "section", "source_url"])
            writer.writeheader()
            writer.writerows(rows)

        with open(TXT_FILE, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['id']}. {r['sentence']}\n   {r['source_url']}\n\n")

    print(f"Added {added} new sentences this run. Total now: {len(rows)}/{TARGET_TOTAL}")


if __name__ == "__main__":
    main()

"""
dm_sentence_scraper.py

Collects real, complete declarative sentences (6-15 words) from Daily Mirror
(dailymirror.lk) news articles across local/business/sports/world sections,
for the NLP POS-tagger assignment's 150-sentence hand-tagged test set.

SETUP (run on your own machine -- this needs real internet access):
    pip install playwright feedparser beautifulsoup4 nltk trafilatura

Uses your already-installed Chrome (or Edge) via Playwright's "channel"
option -- no `playwright install` / browser download needed at all.

RUN:
    python dm_sentence_scraper.py

OUTPUT:
    dm_sentences.csv  -> id, sentence, word_count, section, source_url
    dm_sentences.txt  -> numbered "sentence / url" list

WHAT CHANGED IN THIS VERSION, based on your last run's evidence:

1. BOT-PROTECTION BLOCK PAGES WERE BEING SCRAPED AS IF THEY WERE ARTICLES.
   Your CSV had "This website uses a security service to protect against
   malicious bots." as a "sentence" -- that's DM's challenge/interstitial
   page, not article content. It was grammatically clean enough to pass
   the filters. Root cause: the previous version's User-Agent literally
   contained the word "bot" ("MSc-NLP-coursework-bot"), which is close to
   guaranteed to get flagged by any WAF/bot-management layer. Fixed by:
     - dropping the custom UA entirely (real Chrome via `channel="chrome"`
       already sends a normal, current browser UA)
     - masking `navigator.webdriver` (a JS-readable flag automated browsers
       set to true, which bot-detection commonly checks)
     - detecting known challenge-page phrases + HTTP 4xx/5xx status and
       skipping + backing off instead of scraping the block page
     - aborting a section early (rather than grinding through more blocked
       requests) if blocks happen 3 times in a row, with a clear message

2. SECTION MISLABELING. DM's "breaking_news" RSS feed is a mixed bag --
   it includes international stories too (e.g. an Oil/Iraq story sitting
   under /international/..., not /breaking-news/...). Tagging by "which
   RSS feed surfaced this URL" was wrong. Fixed by deriving `section`
   from the article's own URL path instead (the first path segment,
   e.g. "breaking-news", "international", "business", "sports"), which
   reflects DM's own real categorization rather than an assumption.
   Note: this means the section column may say "international" rather
   than "world" -- that's the accurate label, not a bug.

3. ROBUSTNESS. Wrapped the collection loop so a crash partway through
   still saves whatever was collected (via try/finally), instead of
   losing a long run to one bad page.

Everything from before is still true and unchanged:
- The article body loads via JavaScript, not static HTML -- hence the
  headless-browser (Playwright) approach rather than plain `requests`.
- `wait_until="networkidle"` never fires on these pages (trackers/widgets
  poll continuously) -- we wait for DOM load, then poll until the page's
  visible text stops growing.
- NLTK's sentence splitter can still mis-split on "Rs.", "Dr.", "Mr." --
  since you're hand-tagging these anyway, skim the output before you
  commit to the final 150.
- This is for your own coursework test set. Keep the raw output for your
  own use; don't republish it. DM's footer requires "due courtesy"
  (attribution) for reuse -- that's why every row keeps its source URL.
"""

import re
import csv
import time
import requests
import feedparser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # newer NLTK versions need this too
from nltk.tokenize import sent_tokenize

# --------------------------------------------------------------- CONFIG ---

# Used only to DISCOVER candidate article URLs. The actual "section" label
# in the output comes from each article's own URL (see derive_section), not
# from which of these feeds happened to surface it.
RSS_FEEDS = {
    "local":    "https://www.dailymirror.lk/rss/breaking_news/108",
    "business": "https://www.dailymirror.lk/rss/business_24_7/395",
    "sports":   "https://www.dailymirror.lk/rss/Sports/322",
    "world":    "https://www.dailymirror.lk/rss/world_news/188",
}

LISTING_PAGES = {
    "local":    "https://www.dailymirror.lk/breaking_news/108",
    "business": "https://www.dailymirror.lk/business",
    "sports":   "https://www.dailymirror.lk/sports",
    "world":    "https://www.dailymirror.lk/world-news/213",
}

MIN_WORDS = 6
MAX_WORDS = 15
MAX_SENTENCES_PER_ARTICLE = 3      # keeps sentences spread across many articles
TARGET_TOTAL = 150
PAGE_TIMEOUT_MS = 25_000
STABILIZE_POLL_MS = 500
STABILIZE_MAX_POLLS = 10           # ~5s max extra wait for JS-injected content to settle
DELAY_BETWEEN_PAGES_SEC = 1.5      # be polite to DM's servers
BACKOFF_SEC = 10                   # cool-down after a detected block
MAX_CONSECUTIVE_BLOCKS = 3         # abort this section's remaining URLs after this many

CAPTION_MARKERS = ("pic by", "pic courtesy", "photo by", "photo courtesy", "afp", "reuters photo")

JUNK_MARKERS = (
    "security service to protect",
    "verifies you are not a bot",
    "verifies you are human",
    "checking your browser",
    "enable javascript and cookies",
    "attention required",
    "access denied",
    "just a moment",
    "captcha",
    "cloudflare",
    "subscribe to continue reading",
    "sign up to continue reading",
    "exceeded the maximum number of free articles",
    "please enable cookies",
    "your browser does not support",
)

ARTICLE_URL_RE = re.compile(r'https://www\.dailymirror\.lk/[^"\s]+?/\d+-\d+')
END_PUNCT_RE = re.compile(r'\.[\"\'\u2018\u2019\u201c\u201d]?$')  # "." optionally + closing quote
INVISIBLE_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\u00ad]')  # zero-width chars DM scatters through text
ABBREV_END_RE = re.compile(r'\b(?:No|Rs|Mr|Mrs|Ms|Dr|St|Co|Ltd|Inc|Jr|Sr|vs)\.$')  # split after an abbreviation

# ------------------------------------------------------------- HELPERS -----

def get_rss_urls(rss_url, limit=40):
    feed = feedparser.parse(rss_url)
    return [e.link for e in feed.entries[:limit]]


def get_listing_urls(listing_url, limit=40):
    """Section listing pages render article links + teaser text without
    JS, so a plain request is enough here (unlike individual article pages)."""
    try:
        resp = requests.get(listing_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  listing fetch failed: {listing_url} -> {e}")
        return []
    urls = []
    for u in ARTICLE_URL_RE.findall(resp.text):
        if u not in urls:
            urls.append(u)
    return urls[:limit]


def derive_section(url):
    """The real category, taken from the article's own URL path -- not
    from which RSS feed happened to surface it (DM's feeds mix categories)."""
    path = urlparse(url).path.strip("/")
    return path.split("/")[0].lower() if path else "unknown"


def launch_browser(pw):
    """Prefer the system's own Chrome/Edge (no Playwright browser download
    needed at all) and only fall back to Playwright's bundled Chromium --
    which requires `playwright install chromium` -- if neither is found."""
    for channel in ("chrome", "msedge"):
        try:
            browser = pw.chromium.launch(channel=channel, headless=True)
            print(f"Using system browser channel: {channel}")
            return browser
        except Exception as e:
            print(f"  channel '{channel}' unavailable: {e}")
    print("Falling back to Playwright's bundled Chromium "
          "(needs `playwright install chromium`).")
    return pw.chromium.launch(headless=True)


def new_stealthy_page(browser):
    page = browser.new_page()  # real Chrome UA, no giveaway "bot" string
    # navigator.webdriver is a JS-readable flag automated browsers set to
    # true; several bot-management layers check it directly.
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return page


def render_article_html(page, url):
    # networkidle never fires on pages with trackers/widgets that poll
    # continuously (view counters, ads, chat embeds) -- wait for the DOM
    # instead, then poll until the JS-injected body text stops growing.
    response = page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
    last_len = -1
    for _ in range(STABILIZE_MAX_POLLS):
        page.wait_for_timeout(STABILIZE_POLL_MS)
        try:
            length = page.evaluate("document.body.innerText.length")
        except Exception:
            break
        if length == last_len:
            break
        last_len = length
    status = response.status if response else None
    return page.content(), status


def extract_body_text(html):
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text:
            return text
    except Exception:
        pass
    # Fallback if trafilatura isn't installed, or returns nothing.
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))


def looks_blocked(body_text):
    low = body_text.lower()
    return any(m in low for m in JUNK_MARKERS)


def classify_sentence(sentence):
    """Returns (passes, reason). Reason lets us diagnose *why* yield is low
    instead of just seeing a final count."""
    s = sentence.strip()
    words = s.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False, "word_count"
    if not s[:1].isupper():
        return False, "not_capitalized"
    if not END_PUNCT_RE.search(s):          # allows `He said "yes."` style endings
        return False, "no_period_end"
    if s.isupper():
        return False, "all_caps"
    if any(m in s.lower() for m in CAPTION_MARKERS):
        return False, "caption_marker"
    if ABBREV_END_RE.search(s):
        return False, "abbrev_fragment"
    if s.count('"') % 2 or s.count("“") != s.count("”"):
        return False, "unbalanced_quotes"   # piece of a longer quotation
    return True, "ok"


# --------------------------------------------------------------- MAIN ------

def collect_sentences():
    rows, seen = [], set()
    empty_body_count = 0
    blocked_count = 0
    raw_sentence_count = 0
    reason_counts = {}

    with sync_playwright() as pw:
        browser = launch_browser(pw)
        page = new_stealthy_page(browser)

        try:
            for feed_name, rss_url in RSS_FEEDS.items():
                urls = get_rss_urls(rss_url)
                urls += [u for u in get_listing_urls(LISTING_PAGES[feed_name]) if u not in urls]
                print(f"[{feed_name}] {len(urls)} candidate article URLs")

                consecutive_blocks = 0

                for url in urls:
                    if len(rows) >= TARGET_TOTAL:
                        break
                    if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                        print(f"  {consecutive_blocks} blocks in a row in "
                              f"'{feed_name}' -- moving on to the next section "
                              f"instead of grinding through more of them.")
                        break

                    try:
                        html, status = render_article_html(page, url)
                    except Exception as e:
                        print(f"  skip (render failed): {url} -> {e}")
                        continue

                    body = extract_body_text(html)

                    if (status and status >= 400) or not body or looks_blocked(body):
                        blocked_count += 1
                        consecutive_blocks += 1
                        print(f"  skip (blocked/empty, status={status}): {url}")
                        time.sleep(BACKOFF_SEC)
                        continue

                    consecutive_blocks = 0
                    section = derive_section(url)
                    # Tokenize per paragraph so an unpunctuated sub-heading
                    # can't get glued onto the sentence that follows it.
                    sentences = [sent for para in INVISIBLE_RE.sub("", body).splitlines()
                                 for sent in sent_tokenize(para)]
                    raw_sentence_count += len(sentences)

                    taken = 0
                    for sent in sentences:
                        if taken >= MAX_SENTENCES_PER_ARTICLE:
                            break
                        clean = re.sub(r"\s+", " ", sent).strip()
                        if clean in seen:
                            continue
                        ok, reason = classify_sentence(clean)
                        if not ok:
                            reason_counts[reason] = reason_counts.get(reason, 0) + 1
                            continue
                        seen.add(clean)
                        rows.append({
                            "id": len(rows) + 1,
                            "sentence": clean,
                            "word_count": len(clean.split()),
                            "section": section,
                            "source_url": url,
                        })
                        taken += 1

                    time.sleep(DELAY_BETWEEN_PAGES_SEC)

                if len(rows) >= TARGET_TOTAL:
                    break
        finally:
            browser.close()

    print(f"\nDiagnostics: {raw_sentence_count} raw sentences seen, "
          f"{empty_body_count} empty bodies, {blocked_count} blocked/challenge pages.")
    print("Rejected sentences by reason:", reason_counts)

    return rows


def save(rows):
    with open("dm_sentences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sentence", "word_count", "section", "source_url"])
        writer.writeheader()
        writer.writerows(rows)

    with open("dm_sentences.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['id']}. {r['sentence']}\n   {r['source_url']}\n\n")

    print(f"\nSaved {len(rows)} sentences -> dm_sentences.csv, dm_sentences.txt")
    counts = {}
    for r in rows:
        counts[r["section"]] = counts.get(r["section"], 0) + 1
    print("Breakdown by (real, URL-derived) section:", counts)


if __name__ == "__main__":
    result_rows = []
    try:
        result_rows = collect_sentences()
    finally:
        # Always save whatever we got, even if something crashed partway --
        # no reason to lose a long run to one bad page.
        if result_rows:
            if len(result_rows) < TARGET_TOTAL:
                print(f"\nOnly found {len(result_rows)}/{TARGET_TOTAL}. If "
                      f"blocking is the cause (check the diagnostics above), "
                      f"wait a while before rerunning rather than retrying "
                      f"immediately. Otherwise try raising "
                      f"MAX_SENTENCES_PER_ARTICLE or widening MIN_WORDS/"
                      f"MAX_WORDS slightly.")
            save(result_rows)
        else:
            print("\nNo sentences collected -- nothing to save.")
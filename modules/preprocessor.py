import re
import string
import unicodedata
from typing import Optional

# Cek Sastrawi tersedia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _factory      = StemmerFactory()
    _stemmer      = _factory.create_stemmer()
    _sw_factory   = StopWordRemoverFactory()
    _sw_remover   = _sw_factory.create_stop_word_remover()
    SASTRAWI_OK   = True
except ImportError:
    SASTRAWI_OK   = False


# ─── Stopword list tambahan manual ────────────────────────────────────────────
EXTRA_STOPWORDS = {
    "yg", "yak", "gak", "nggak", "kalo", "tapi", "juga", "sudah", "sudah",
    "udah", "aja", "sih", "deh", "dong", "lho", "lah", "nih", "wah",
    "adalah", "merupakan", "tersebut", "itu", "ini", "dengan", "yang",
    "dan", "di", "ke", "dari", "atau", "pada", "oleh", "untuk", "dalam",
    "telah", "akan", "dapat", "bisa", "ada", "tidak", "tak", "bukan",
}


def normalize_unicode(text: str) -> str:
    """Normalisasi karakter unicode (hapus karakter aneh)."""
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_mentions_hashtags(text: str) -> str:
    return re.sub(r"[@#]\w+", " ", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_punctuation(text: str) -> str:
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def case_fold(text: str) -> str:
    return text.lower()


def remove_stopwords_manual(text: str) -> str:
    """Stopword removal manual tanpa Sastrawi (fallback)."""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in EXTRA_STOPWORDS)


def stem_text(text: str) -> str:
    if SASTRAWI_OK:
        return _stemmer.stem(text)
    return text


def remove_stopwords_sastrawi(text: str) -> str:
    if SASTRAWI_OK:
        return _sw_remover.remove(text)
    return remove_stopwords_manual(text)


def clean_text(
    text: str,
    remove_stop: bool = True,
    do_stem: bool = True,
    remove_nums: bool = True,
) -> str:
    """
    Pipeline preprocessing teks lengkap:
    unicode → lowercase → hapus URL/mention/HTML → hapus angka →
    hapus tanda baca → stopword → stemming → whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalize_unicode(text)
    text = case_fold(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_html_tags(text)
    if remove_nums:
        text = remove_numbers(text)
    text = remove_punctuation(text)
    if remove_stop:
        text = remove_stopwords_sastrawi(text)
        # Juga hapus stopword ekstra manual
        text = remove_stopwords_manual(text)
    if do_stem:
        text = stem_text(text)
    text = remove_extra_whitespace(text)
    return text


def preprocess_dataframe(df, text_col: str = "text", new_col: str = "clean_text",
                          remove_stop: bool = True, do_stem: bool = True) -> "pd.DataFrame":
    """Terapkan clean_text ke seluruh kolom DataFrame."""
    import pandas as pd
    df = df.copy()
    df[new_col] = df[text_col].apply(
        lambda t: clean_text(str(t), remove_stop=remove_stop, do_stem=do_stem)
    )
    return df


if __name__ == "__main__":
    sample = """
    BREAKING: Vaksin Covid-19 mengandung chip mikro! Ini buktinya https://t.co/abc123
    @narasumber #faktaatauhoaks Pemerintah tidak mau memberitahu rakyatnya!!!
    """
    print("Original :", sample[:80])
    print("Cleaned  :", clean_text(sample))

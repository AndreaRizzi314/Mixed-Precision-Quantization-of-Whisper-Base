"""Word edit scoring with the normalization used in the recorded experiment."""
import unicodedata


def normalize(text: str) -> str:
    """Uppercase, strip punctuation without splitting contractions, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).upper()
    return " ".join("".join(c for c in text if c.isalnum() or c.isspace()).split())

def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word edits and reference word count."""
    a, b = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        new = [i]
        for j, predicted in enumerate(b, 1):
            new.append(min(row[j] + 1, new[-1] + 1, row[j - 1] + (word != predicted)))
        row = new
    return row[-1], len(a)

def corpus_wer(pairs: list[tuple[str, str]]) -> dict:
    counts = [word_errors(ref, hyp) for ref, hyp in pairs]
    edits = sum(x for x, _ in counts)
    words = sum(n for _, n in counts)
    if not words:
        raise ValueError("References contain no words")
    return {"wer": edits / words, "errors": edits, "reference_words": words,
            "clip_wer": [x / n if n else None for x, n in counts]}

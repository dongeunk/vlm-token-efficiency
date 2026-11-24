"""Study4 scoring functions, copied unchanged."""
from collections import Counter
import re


def parse_answer(text):
    # Answer-only contract paired with an explicit yes/no-only prompt.
    if not isinstance(text, str):
        return "invalid"
    match = re.fullmatch(r"(yes|no)[.!]?", text.strip(), re.IGNORECASE | re.ASCII)
    return match.group(1).lower() if match else "invalid"


def metrics(results):
    counts = Counter((r["label"], r["prediction"]) for r in results)
    tp, tn = counts["yes", "yes"], counts["no", "no"]
    fp, fn = counts["no", "yes"], counts["yes", "no"]
    iy, ino = counts["yes", "invalid"], counts["no", "invalid"]
    n = tp + tn + fp + fn + iy + ino

    def ratio(a, b):
        return a / b if b else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn + iy)
    return dict(n=n, accuracy=ratio(tp + tn, n), precision=precision, recall=recall,
                f1=ratio(2 * precision * recall, precision + recall),
                invalid_rate=ratio(iy + ino, n), tp=tp, fp=fp, tn=tn, fn=fn,
                invalid_yes=iy, invalid_no=ino)


def official_answer(raw):
    if not isinstance(raw, str):
        raise TypeError("Official scoring expects the saved response string")
    words = raw.split(".", 1)[0].replace(",", "").split(" ")
    return "no" if {"No", "no", "not"}.intersection(words) else "yes"

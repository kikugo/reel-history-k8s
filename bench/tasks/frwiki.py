"""process_results for the French Wikipedia perplexity task (same counting as lm-eval's wikitext)."""

import re


def process_results(doc, results):
    (loglikelihood,) = results
    words = len(re.split(r"\s+", doc["page"]))
    n_bytes = len(doc["page"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, n_bytes),
        "bits_per_byte": (loglikelihood, n_bytes),
    }

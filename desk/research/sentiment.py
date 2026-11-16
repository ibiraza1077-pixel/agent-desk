"""Headline sentiment with FinBERT, run locally (Apple GPU via MPS when available)."""
from __future__ import annotations

import logging
import sqlite3
import time

log = logging.getLogger(__name__)

MAX_TOKENS = 64  # headlines are short; longer ones are truncated
COMMIT_EVERY = 10  # batches


def score_news(con: sqlite3.Connection, model_name: str = "ProsusAI/finbert", batch_size: int = 64) -> int:
    """Score every news headline that has no news_sentiment row for `model_name`. Returns the number scored."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logging.getLogger("httpx").setLevel(logging.WARNING)  # transformers re-enables noisy per-file HTTP logs
    rows = con.execute("SELECT id, headline FROM news WHERE id NOT IN (SELECT id FROM news_sentiment WHERE model=?)",
                       (model_name,)).fetchall()
    if not rows:
        log.info("sentiment: nothing to score for %s", model_name)
        return 0
    # Similar-length headlines per batch -> less padding -> faster.
    rows.sort(key=lambda r: len(r[1]))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # Read the label order from the model rather than assuming indices.
    label_idx = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    try:
        i_pos, i_neg, i_neu = label_idx["positive"], label_idx["negative"], label_idx["neutral"]
    except KeyError as e:
        raise RuntimeError(f"{model_name} labels {sorted(label_idx)} lack positive/negative/neutral") from e
    log.info("sentiment: scoring %d headlines with %s on %s", len(rows), model_name, device)

    t0 = time.monotonic()
    done = 0
    with torch.inference_mode():
        for n, i in enumerate(range(0, len(rows), batch_size), 1):
            chunk = rows[i:i + batch_size]
            enc = tok([h for _, h in chunk], truncation=True, max_length=MAX_TOKENS, padding=True,
                      return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().tolist()
            con.executemany(
                "INSERT OR REPLACE INTO news_sentiment(id, model, p_pos, p_neg, p_neu, score) VALUES(?,?,?,?,?,?)",
                [(r[0], model_name, p[i_pos], p[i_neg], p[i_neu], p[i_pos] - p[i_neg]) for r, p in zip(chunk, probs)])
            done += len(chunk)
            if n % COMMIT_EVERY == 0:
                con.commit()
                log.info("sentiment: %d/%d (%.0f headlines/s)", done, len(rows), done / (time.monotonic() - t0))
    con.commit()
    log.info("sentiment: scored %d headlines in %.1fs (%.0f headlines/s)", done, time.monotonic() - t0,
             done / max(time.monotonic() - t0, 1e-9))
    return done

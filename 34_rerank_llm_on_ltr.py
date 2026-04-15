"""
LLM-based reranker on top of LTR+CE predictions (script 24).

Pointwise relevance scoring with a small instruct LLM (Qwen2.5-1.5B-Instruct
by default, ~3GB, runs on MPS/CPU). The LLM is asked to emit a relevance
score 0-10 for each (query, candidate) pair. Score is extracted from the
first numeric token of the generated answer; if parsing fails we fall back
to the logit-based expectation over "0".."10".

Alternative backend: set OPENAI_API_KEY and pass --backend openai to use a
cheap OpenAI model (default gpt-4o-mini). The prompt is identical.

The reranker rescores only the top-K candidates from script 24 and keeps the
tail untouched, then interpolates with the LTR rank position (gamma).

Usage:
  python3 34_rerank_llm_on_ltr.py                      # local, top-20
  python3 34_rerank_llm_on_ltr.py --rerank-top 30
  python3 34_rerank_llm_on_ltr.py --backend openai --llm-model gpt-4o-mini
  python3 34_rerank_llm_on_ltr.py --submit-held-out
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (evaluate, format_text, get_device, load_corpus,
                   load_qrels, load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"

DEFAULT_LTR_SUB = SCRIPT_DIR / "submissions" / "ltr_ce" / "submission_data.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "rerank_llm_on_ltr"

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_RERANK_TOP = 20
DEFAULT_GAMMA = 0.6
DEFAULT_MAX_QUERY_CHARS = 800
DEFAULT_MAX_DOC_CHARS = 800

PROMPT_SYSTEM = (
    "You are an expert at judging scientific citation relevance. "
    "Given a query paper and a candidate paper, rate how likely the candidate "
    "is cited by the query paper on a scale from 0 (unrelated) to 10 "
    "(definitely cited). Answer with a single integer only."
)

PROMPT_USER_TEMPLATE = (
    "Query paper:\n{query}\n\n"
    "Candidate paper:\n{doc}\n\n"
    "Relevance score (0-10):"
)


def build_user_prompt(q_text: str, d_text: str, max_q: int, max_d: int) -> str:
    return PROMPT_USER_TEMPLATE.format(
        query=q_text[:max_q].strip(),
        doc=d_text[:max_d].strip(),
    )


def parse_score_from_text(text: str) -> float | None:
    m = re.search(r"\b(10|[0-9])\b", text)
    if m is None:
        return None
    return float(m.group(1))


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ── Local HF backend ──────────────────────────────────────────────────────

class LocalLLM:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        print(f"Loading local LLM {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        self.model.eval()

        self.digit_token_ids = []
        for s in [str(i) for i in range(11)]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            self.digit_token_ids.append(ids[0] if ids else -1)

    def score(self, system_prompt: str, user_prompts: list[str]) -> np.ndarray:
        import torch
        scores = np.zeros(len(user_prompts), dtype=np.float32)
        for i, up in enumerate(user_prompts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)
            last_logits = out.logits[0, -1]  # (vocab,)

            digit_logits = []
            for tid in self.digit_token_ids:
                digit_logits.append(
                    last_logits[tid].item() if tid >= 0 else -1e9
                )
            digit_logits = np.array(digit_logits, dtype=np.float32)
            probs = np.exp(digit_logits - digit_logits.max())
            probs /= probs.sum()
            # Expected value over 0..10 → smooth score
            scores[i] = float(np.dot(probs, np.arange(11, dtype=np.float32)))
        return scores


# ── OpenAI backend ────────────────────────────────────────────────────────

class OpenAILLM:
    def __init__(self, model_name: str):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI()
        self.model = model_name

    def score(self, system_prompt: str, user_prompts: list[str]) -> np.ndarray:
        scores = np.zeros(len(user_prompts), dtype=np.float32)
        for i, up in enumerate(user_prompts):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": up},
                ],
                temperature=0.0,
                max_tokens=4,
            )
            text = resp.choices[0].message.content or ""
            parsed = parse_score_from_text(text)
            scores[i] = parsed if parsed is not None else 5.0
        return scores


def main():
    parser = argparse.ArgumentParser(
        description="LLM pointwise reranker on top of LTR+CE predictions"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", default=DEFAULT_LTR_SUB, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--llm-model", default=None,
                        help="Override default model for the chosen backend")
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
                        help="Number of top LTR candidates to rerank (LLM is slow)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="Weight on LLM score vs LTR rank")
    parser.add_argument("--max-query-chars", type=int, default=DEFAULT_MAX_QUERY_CHARS)
    parser.add_argument("--max-doc-chars", type=int, default=DEFAULT_MAX_DOC_CHARS)
    parser.add_argument("--submit-held-out", action="store_true")
    parser.add_argument("--limit-queries", type=int, default=0,
                        help="Debug: only rerank the first N queries")
    args = parser.parse_args()

    is_heldout = args.submit_held_out

    if args.backend == "local":
        model_name = args.llm_model or DEFAULT_LOCAL_MODEL
        device = get_device()
        print(f"Backend: local  |  Model: {model_name}  |  Device: {device}")
    else:
        model_name = args.llm_model or DEFAULT_OPENAI_MODEL
        print(f"Backend: openai |  Model: {model_name}")

    print(f"Rerank top-{args.rerank_top}, gamma={args.gamma}")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: LTR predictions not found at {pred_path}")
        print("Run script 24 first (with --submit-held-out if needed).")
        return
    with open(pred_path) as f:
        ltr_preds = json.load(f)
    print(f"Loaded {len(ltr_preds)} LTR rankings from {pred_path}")

    print("Loading corpus / queries...")
    corpus = load_corpus(args.corpus)
    corpus_texts = {row["doc_id"]: format_text(row) for _, row in corpus.iterrows()}

    queries = load_queries(args.held_out if is_heldout else args.queries)
    query_ids = queries["doc_id"].tolist()
    query_texts = {row["doc_id"]: format_text(row) for _, row in queries.iterrows()}

    if args.limit_queries > 0:
        query_ids = query_ids[: args.limit_queries]
        print(f"  DEBUG: limited to {len(query_ids)} queries")

    if args.backend == "local":
        llm = LocalLLM(model_name, get_device())
    else:
        llm = OpenAILLM(model_name)

    predictions = {}
    for qid in tqdm(query_ids, desc="LLM rerank"):
        ltr_ranked = ltr_preds.get(qid, [])
        if not ltr_ranked:
            predictions[qid] = []
            continue

        top_candidates = ltr_ranked[: args.rerank_top]
        tail = ltr_ranked[args.rerank_top:]
        q_text = query_texts.get(qid, "")

        user_prompts = [
            build_user_prompt(
                q_text, corpus_texts.get(d, ""),
                args.max_query_chars, args.max_doc_chars,
            )
            for d in top_candidates
        ]

        llm_scores = llm.score(PROMPT_SYSTEM, user_prompts)

        ltr_rank_score = np.array(
            [1.0 / (60 + r + 1) for r in range(len(top_candidates))],
            dtype=np.float32,
        )
        final = (args.gamma * normalize_minmax(llm_scores)
                 + (1.0 - args.gamma) * normalize_minmax(ltr_rank_score))

        order = np.argsort(-final)
        reranked_top = [top_candidates[i] for i in order]

        merged = reranked_top + tail
        seen = set()
        deduped = []
        for d in merged:
            if d not in seen:
                deduped.append(d)
                seen.add(d)
        predictions[qid] = deduped[:100]

    if not is_heldout and args.limit_queries == 0:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()

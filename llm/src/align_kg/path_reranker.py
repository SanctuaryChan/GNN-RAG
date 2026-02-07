import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CrossEncoderReranker(object):
    def __init__(self, model_path, device=None, max_length=256, use_answer=True, use_question=True):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_answer = use_answer
        self.use_question = use_question

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_text(self, question, path_text, answer=None):
        parts = []
        if self.use_question and question:
            parts.append(f"[Q] {question}")
        parts.append(f"[PATH] {path_text}")
        if self.use_answer and answer:
            parts.append(f"[ANS] {answer}")
        return " ".join(parts)

    @torch.no_grad()
    def score(self, question, path_texts, answers=None, rel_paths=None):
        if answers is None or len(answers) != len(path_texts):
            answers = [None] * len(path_texts)
        texts = [self._build_text(question, p, a) for p, a in zip(path_texts, answers)]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        logits = outputs.logits.squeeze(-1)
        scores = logits.detach().cpu().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        return scores

    def rerank(self, question, path_texts, answers=None, rel_paths=None, topk=5):
        if not path_texts:
            return []
        scores = self.score(question, path_texts, answers=answers, rel_paths=rel_paths)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        topk = min(topk, len(path_texts))
        return [path_texts[i] for i, _ in indexed[:topk]]


class AlignReranker(object):
    def __init__(self, model_dir, device=None, max_length=256, use_answer=True, use_question=True):
        from align_kg.graph_text_ranker import GraphTextRanker

        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_answer = use_answer
        self.use_question = use_question

        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.model = GraphTextRanker(
            cfg["model_name_or_path"],
            share_encoder=cfg.get("share_encoder", True),
        )
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def _build_text(self, question, path_text, answer=None, prefix="PATH"):
        parts = []
        if self.use_question and question:
            parts.append(f"[Q] {question}")
        parts.append(f"[{prefix}] {path_text}")
        if self.use_answer and answer:
            parts.append(f"[ANS] {answer}")
        return " ".join(parts)

    @torch.no_grad()
    def score(self, question, path_texts, answers=None, rel_paths=None):
        if answers is None or len(answers) != len(path_texts):
            answers = [None] * len(path_texts)
        if rel_paths is None or len(rel_paths) != len(path_texts):
            rel_paths = [[] for _ in path_texts]
        graph_texts = [
            self._build_text(question, " / ".join(rp), ans, prefix="REL")
            for rp, ans in zip(rel_paths, answers)
        ]
        path_texts = [
            self._build_text(question, p, ans, prefix="PATH")
            for p, ans in zip(path_texts, answers)
        ]
        g_enc = self.tokenizer(
            graph_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        t_enc = self.tokenizer(
            path_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        g_enc = {k: v.to(self.device) for k, v in g_enc.items()}
        t_enc = {k: v.to(self.device) for k, v in t_enc.items()}
        scores, _ = self.model(g_enc, t_enc, labels=None, align=False)
        scores = scores.detach().cpu().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        return scores

    def rerank(self, question, path_texts, answers=None, rel_paths=None, topk=5):
        if not path_texts:
            return []
        scores = self.score(question, path_texts, answers=answers, rel_paths=rel_paths)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        topk = min(topk, len(path_texts))
        return [path_texts[i] for i, _ in indexed[:topk]]


def build_reranker(model_path, mode="cross", **kwargs):
    if mode == "cross":
        return CrossEncoderReranker(model_path, **kwargs)
    if mode == "align":
        return AlignReranker(model_path, **kwargs)
    raise ValueError(f"Unsupported reranker mode: {mode}")

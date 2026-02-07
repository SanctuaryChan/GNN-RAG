import os
import json
import argparse
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from align_kg.graph_text_ranker import GraphTextRanker


class PathRankDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        self.items = []
        with open(data_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(obj)
                if max_samples and len(self.items) >= max_samples:
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_text(question, path_text, answer=None, use_answer=True, use_question=True, prefix="PATH"):
    parts = []
    if use_question and question:
        parts.append(f"[Q] {question}")
    parts.append(f"[{prefix}] {path_text}")
    if use_answer and answer:
        parts.append(f"[ANS] {answer}")
    return " ".join(parts)


def collate_cross(batch, tokenizer, max_length, use_answer, use_question):
    texts = []
    labels = []
    for item in batch:
        texts.append(build_text(item.get('question'), item.get('path_text'), item.get('answer'), use_answer, use_question))
        labels.append(item.get('label', 0))
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.float)
    return enc, labels


def collate_align(batch, tokenizer, max_length, use_answer, use_question):
    graph_texts = []
    path_texts = []
    labels = []
    for item in batch:
        rel_path = item.get('rel_path', [])
        rel_text = " / ".join(rel_path)
        graph_texts.append(build_text(item.get('question'), rel_text, item.get('answer'), use_answer, use_question, prefix="REL"))
        path_texts.append(build_text(item.get('question'), item.get('path_text'), item.get('answer'), use_answer, use_question, prefix="PATH"))
        labels.append(item.get('label', 0))
    graph_enc = tokenizer(graph_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    path_enc = tokenizer(path_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.float)
    return graph_enc, path_enc, labels


def save_model_cross(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_align(model, tokenizer, output_dir, model_name_or_path, share_encoder):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    cfg = {"model_name_or_path": model_name_or_path, "share_encoder": share_encoder}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    tokenizer.save_pretrained(output_dir)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--mode", type=str, default="cross", choices=["cross", "align"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--share_encoder", action="store_true")
    parser.add_argument("--use_answer", action="store_true")
    parser.add_argument("--use_question", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    dataset = PathRankDataset(args.data_path, max_samples=args.max_samples)

    if args.mode == "cross":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1)
        model.to(device)
        model.train()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_cross(b, tokenizer, args.max_length, args.use_answer, args.use_question),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_steps = args.epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_steps * args.warmup_ratio),
            num_training_steps=num_steps,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            for enc, labels in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                outputs = model(**enc)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        save_model_cross(model, tokenizer, args.output_dir)
        return

    # align mode
    model = GraphTextRanker(args.model_name_or_path, share_encoder=args.share_encoder)
    model.to(device)
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_align(b, tokenizer, args.max_length, args.use_answer, args.use_question),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_steps * args.warmup_ratio),
        num_training_steps=num_steps,
    )

    for epoch in range(args.epochs):
        for graph_enc, path_enc, labels in loader:
            graph_enc = {k: v.to(device) for k, v in graph_enc.items()}
            path_enc = {k: v.to(device) for k, v in path_enc.items()}
            labels = labels.to(device)
            _, loss = model(graph_enc, path_enc, labels=labels, align=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    save_model_align(model, tokenizer, args.output_dir, args.model_name_or_path, args.share_encoder)


if __name__ == "__main__":
    train()

import os
import json
import argparse
import random
from collections import defaultdict

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


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def build_text(question, path_text, answer=None, use_answer=True, use_question=True, prefix="PATH"):
    parts = []
    if use_question and question:
        parts.append(f"[Q] {question}")
    parts.append(f"[{prefix}] {path_text}")
    if use_answer and answer:
        parts.append(f"[ANS] {answer}")
    return " ".join(parts)


def collate_cross(batch, tokenizer, max_length, use_answer, use_question, return_questions=False):
    texts = []
    labels = []
    questions = []
    for item in batch:
        texts.append(build_text(item.get('question'), item.get('path_text'), item.get('answer'), use_answer, use_question))
        labels.append(item.get('label', 0))
        if return_questions:
            questions.append(item.get('question', ''))
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.float)
    if return_questions:
        return enc, labels, questions
    return enc, labels


def collate_align(batch, tokenizer, max_length, use_answer, use_question, return_questions=False):
    graph_texts = []
    path_texts = []
    labels = []
    questions = []
    for item in batch:
        rel_path = item.get('rel_path', [])
        rel_text = " / ".join(rel_path)
        graph_texts.append(build_text(item.get('question'), rel_text, item.get('answer'), use_answer, use_question, prefix="REL"))
        path_texts.append(build_text(item.get('question'), item.get('path_text'), item.get('answer'), use_answer, use_question, prefix="PATH"))
        labels.append(item.get('label', 0))
        if return_questions:
            questions.append(item.get('question', ''))
    graph_enc = tokenizer(graph_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    path_enc = tokenizer(path_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.float)
    if return_questions:
        return graph_enc, path_enc, labels, questions
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


def split_by_question(items, eval_ratio, seed):
    q_to_indices = defaultdict(list)
    for idx, item in enumerate(items):
        q_to_indices[item.get("question", "")].append(idx)
    questions = list(q_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(questions)
    n_eval = max(1, int(len(questions) * eval_ratio))
    eval_q = set(questions[:n_eval])
    train_indices = []
    eval_indices = []
    for q, idxs in q_to_indices.items():
        if q in eval_q:
            eval_indices.extend(idxs)
        else:
            train_indices.extend(idxs)
    return train_indices, eval_indices


def compute_ranking_metrics(scores, labels, questions):
    q_to_items = defaultdict(list)
    for score, label, q in zip(scores, labels, questions):
        q_to_items[q].append((score, label))

    hit1_total = 0
    mrr_total = 0.0
    q_count = 0
    for q, items in q_to_items.items():
        items.sort(key=lambda x: x[0], reverse=True)
        q_count += 1
        hit1_total += 1 if items and items[0][1] == 1 else 0
        rank = None
        for idx, (_, label) in enumerate(items, start=1):
            if label == 1:
                rank = idx
                break
        if rank is not None:
            mrr_total += 1.0 / rank
    if q_count == 0:
        return {"hit1": 0.0, "mrr": 0.0}
    return {"hit1": hit1_total / q_count, "mrr": mrr_total / q_count}


def evaluate_cross(model, loader, device):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    all_scores = []
    all_labels = []
    all_questions = []
    with torch.no_grad():
        for enc, labels, questions in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            outputs = model(**enc)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_count += labels.numel()
            all_scores.extend(logits.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_questions.extend(questions)
    avg_loss = total_loss / max(1, total_count)
    metrics = compute_ranking_metrics(all_scores, all_labels, all_questions)
    metrics["loss"] = avg_loss
    return metrics


def evaluate_align(model, loader, device):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    all_scores = []
    all_labels = []
    all_questions = []
    with torch.no_grad():
        for graph_enc, path_enc, labels, questions in loader:
            graph_enc = {k: v.to(device) for k, v in graph_enc.items()}
            path_enc = {k: v.to(device) for k, v in path_enc.items()}
            labels = labels.to(device)
            scores, _ = model(graph_enc, path_enc, labels=None, align=False)
            loss = loss_fn(scores, labels)
            total_loss += loss.item()
            total_count += labels.numel()
            all_scores.extend(scores.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_questions.extend(questions)
    avg_loss = total_loss / max(1, total_count)
    metrics = compute_ranking_metrics(all_scores, all_labels, all_questions)
    metrics["loss"] = avg_loss
    return metrics


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
    parser.add_argument("--eval_path", type=str, default=None, help="optional eval dataset path")
    parser.add_argument("--eval_ratio", type=float, default=0.0, help="split ratio from train if eval_path is not set")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0, help="early stop patience (0 to disable)")
    parser.add_argument("--early_stop_metric", type=str, default="mrr", choices=["loss", "hit1", "mrr"])
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--save_each_epoch", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    dataset = PathRankDataset(args.data_path, max_samples=args.max_samples)
    eval_dataset = None
    train_dataset = dataset
    if args.eval_path:
        eval_dataset = PathRankDataset(args.eval_path, max_samples=None)
    elif args.eval_ratio and args.eval_ratio > 0:
        train_indices, eval_indices = split_by_question(dataset.items, args.eval_ratio, args.eval_seed)
        train_dataset = SubsetDataset(dataset, train_indices)
        eval_dataset = SubsetDataset(dataset, eval_indices)

    if args.mode == "cross":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1)
        model.to(device)
        model.train()
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_cross(b, tokenizer, args.max_length, args.use_answer, args.use_question),
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_cross(b, tokenizer, args.max_length, args.use_answer, args.use_question, return_questions=True),
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_steps = args.epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_steps * args.warmup_ratio),
            num_training_steps=num_steps,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_score = None
        bad_epochs = 0
        higher_is_better = args.early_stop_metric in {"hit1", "mrr"}
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

            if args.save_each_epoch:
                save_model_cross(model, tokenizer, os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}"))

            if eval_loader is not None:
                metrics = evaluate_cross(model, eval_loader, device)
                metric_value = metrics[args.early_stop_metric]
                print(f"[Eval] epoch={epoch+1} loss={metrics['loss']:.6f} hit1={metrics['hit1']:.4f} mrr={metrics['mrr']:.4f}")

                improved = False
                if best_score is None:
                    improved = True
                elif higher_is_better:
                    improved = metric_value > (best_score + args.min_delta)
                else:
                    improved = metric_value < (best_score - args.min_delta)

                if improved:
                    best_score = metric_value
                    bad_epochs = 0
                    if args.save_best:
                        save_model_cross(model, tokenizer, os.path.join(args.output_dir, "best"))
                else:
                    bad_epochs += 1
                    if args.patience > 0 and bad_epochs >= args.patience:
                        print(f"Early stopping at epoch {epoch+1} (best {args.early_stop_metric}={best_score:.6f})")
                        break

            model.train()

        save_model_cross(model, tokenizer, args.output_dir)
        return

    # align mode
    model = GraphTextRanker(args.model_name_or_path, share_encoder=args.share_encoder)
    model.to(device)
    model.train()
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_align(b, tokenizer, args.max_length, args.use_answer, args.use_question),
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_align(b, tokenizer, args.max_length, args.use_answer, args.use_question, return_questions=True),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_steps * args.warmup_ratio),
        num_training_steps=num_steps,
    )

    best_score = None
    bad_epochs = 0
    higher_is_better = args.early_stop_metric in {"hit1", "mrr"}
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

        if args.save_each_epoch:
            save_model_align(model, tokenizer, os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}"), args.model_name_or_path, args.share_encoder)

        if eval_loader is not None:
            metrics = evaluate_align(model, eval_loader, device)
            metric_value = metrics[args.early_stop_metric]
            print(f"[Eval] epoch={epoch+1} loss={metrics['loss']:.6f} hit1={metrics['hit1']:.4f} mrr={metrics['mrr']:.4f}")

            improved = False
            if best_score is None:
                improved = True
            elif higher_is_better:
                improved = metric_value > (best_score + args.min_delta)
            else:
                improved = metric_value < (best_score - args.min_delta)

            if improved:
                best_score = metric_value
                bad_epochs = 0
                if args.save_best:
                    save_model_align(model, tokenizer, os.path.join(args.output_dir, "best"), args.model_name_or_path, args.share_encoder)
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch+1} (best {args.early_stop_metric}={best_score:.6f})")
                    break

        model.train()

    save_model_align(model, tokenizer, args.output_dir, args.model_name_or_path, args.share_encoder)


if __name__ == "__main__":
    train()

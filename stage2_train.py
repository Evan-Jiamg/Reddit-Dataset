"""
Stage 2: RoBERTa Stance Model Training
Load IBM Argument Quality Ranking 30K (HuggingFace) →
fine-tune RoBERTa-base (Linear(768→1) + Tanh) with MSE loss →
4-fold cross-topic validation → save best checkpoint

Dataset: ibm-research/argument_quality_ranking_30k
Fields used: argument, topic, stance_WA (-1/+1), stance_WA_conf
"""

import os
import math
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "model")


# ── Dataset ────────────────────────────────────────────────────────────────
class StanceDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["topic"],
            s["argument"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stance": torch.tensor(s["stance"], dtype=torch.float32),
        }


# ── Model ──────────────────────────────────────────────────────────────────
class StanceModel(nn.Module):
    def __init__(self, roberta_name: str = "roberta-base"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.regressor = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.tanh(self.regressor(cls)).squeeze(-1)


# ── Training loop ──────────────────────────────────────────────────────────
def train_fold(model, train_loader, val_loader, device, epochs: int, lr: float):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    criterion = nn.MSELoss()
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stance = batch["stance"].to(device)

            optimizer.zero_grad()
            pred = model(input_ids, attention_mask)
            loss = criterion(pred, stance)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                ).cpu().numpy()
                val_preds.extend(pred.tolist())
                val_labels.extend(batch["stance"].numpy().tolist())

        val_mse = float(np.mean((np.array(val_preds) - np.array(val_labels)) ** 2))
        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_labels))))
        print(f"  Epoch {epoch}/{epochs}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_mse


# ── Main ───────────────────────────────────────────────────────────────────
def run(epochs: int, lr: float, batch_size: int, max_length: int,
        roberta_name: str, n_folds: int, min_conf: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Load dataset
    print("Loading ibm-research/argument_quality_ranking_30k ...")
    raw = load_dataset(
        "ibm-research/argument_quality_ranking_30k",
        "argument_quality_ranking",
    )
    # Merge all splits
    all_rows = []
    for split in raw.values():
        for row in split:
            conf = float(row["stance_WA_conf"])
            if conf < min_conf:
                continue
            all_rows.append({
                "topic": row["topic"],
                "argument": row["argument"],
                "stance": float(row["stance_WA"]),  # -1.0 or +1.0
            })

    print(f"  Loaded {len(all_rows):,} examples "
          f"(conf >= {min_conf}, "
          f"{len(set(r['topic'] for r in all_rows))} topics)")

    topics = [r["topic"] for r in all_rows]
    unique_topics = sorted(set(topics))
    topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_idx[t] for t in topics])

    tokenizer = RobertaTokenizer.from_pretrained(roberta_name)

    # 2. Cross-topic k-fold
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(all_rows, groups=groups), 1
    ):
        val_topics = set(unique_topics[groups[i]] for i in val_idx)
        print(f"\n--- Fold {fold}/{n_folds}  "
              f"({len(val_topics)} val topics, {len(val_idx)} examples) ---")

        train_samples = [all_rows[i] for i in train_idx]
        val_samples = [all_rows[i] for i in val_idx]

        train_loader = DataLoader(
            StanceDataset(train_samples, tokenizer, max_length),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            StanceDataset(val_samples, tokenizer, max_length),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        model = StanceModel(roberta_name).to(device)
        best_state, best_val_mse = train_fold(
            model, train_loader, val_loader, device, epochs, lr
        )
        fold_results.append(best_val_mse)

        ckpt_path = os.path.join(
            CKPT_DIR, f"fold{fold}_mse{best_val_mse:.4f}.pt"
        )
        torch.save({
            "model_state": best_state,
            "roberta_name": roberta_name,
            "val_mse": best_val_mse,
            "fold": fold,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # 3. Summary
    mean_mse = float(np.mean(fold_results))
    print(f"\n=== Cross-topic {n_folds}-fold results ===")
    for i, mse in enumerate(fold_results, 1):
        print(f"  Fold {i}: MSE={mse:.4f}  RMSE={math.sqrt(mse):.4f}")
    print(f"  Mean MSE={mean_mse:.4f}  Mean RMSE={math.sqrt(mean_mse):.4f}")

    # 4. Final model on all data
    print("\nTraining final model on all data...")
    full_loader = DataLoader(
        StanceDataset(all_rows, tokenizer, max_length),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    final_model = StanceModel(roberta_name).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        final_model.train()
        total_loss = 0.0
        for batch in full_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stance = batch["stance"].to(device)
            optimizer.zero_grad()
            pred = final_model(input_ids, attention_mask)
            loss = criterion(pred, stance)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  Final epoch {epoch}/{epochs}  "
              f"loss={total_loss/len(full_loader):.4f}")

    final_path = os.path.join(CKPT_DIR, "final_model.pt")
    torch.save({
        "model_state": {k: v.cpu() for k, v in final_model.state_dict().items()},
        "roberta_name": roberta_name,
        "cv_mean_mse": mean_mse,
        "cv_fold_mses": fold_results,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    cv_path = os.path.join(CKPT_DIR, "cv_results.json")
    with open(cv_path, "w") as f:
        json.dump({
            "fold_mses": fold_results,
            "mean_mse": mean_mse,
            "mean_rmse": math.sqrt(mean_mse),
            "dataset": "ibm-research/argument_quality_ranking_30k",
            "n_examples": len(all_rows),
            "n_topics": len(unique_topics),
        }, f, indent=2)
    print(f"CV results saved: {cv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: RoBERTa stance model training"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--roberta", default="roberta-base")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=0.8,
                        help="Minimum stance_WA_conf threshold (default: 0.8)")
    args = parser.parse_args()

    run(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        roberta_name=args.roberta,
        n_folds=args.folds,
        min_conf=args.min_conf,
    )

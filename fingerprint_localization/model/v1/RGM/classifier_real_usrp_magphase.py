from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset


class LocationClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stratified_split(labels: np.ndarray, valid_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    counts = np.bincount(labels)
    stratify = labels if np.min(counts[counts > 0]) >= 3 else None

    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed,
        stratify=stratify,
    )
    train_valid_labels = labels[train_valid_idx]
    counts_tv = np.bincount(train_valid_labels)
    stratify_tv = train_valid_labels if np.min(counts_tv[counts_tv > 0]) >= 2 else None
    valid_ratio = valid_frac / max(1.0 - test_frac, 1e-8)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_ratio,
        random_state=seed,
        stratify=stratify_tv,
    )
    return train_idx, valid_idx, test_idx


def make_loader(features: torch.Tensor, labels: torch.Tensor, indices: Iterable[int], batch_size: int, shuffle: bool) -> DataLoader:
    idx = torch.tensor(list(indices), dtype=torch.long)
    dataset = TensorDataset(features[idx], labels[idx])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    return accuracy_score(y_true, y_pred), y_true, y_pred


def run_knn(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, k: int) -> dict:
    k = min(k, len(train_idx))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(features[train_idx], labels[train_idx])
    pred = clf.predict(features[test_idx])
    true = labels[test_idx]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(true, pred, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real-data classifier on USRP 2x16 LoRa mag/phase features.")
    parser.add_argument("--dataset", type=Path, default=Path("model/v1/input/usrp_magphase16_real_dataset.pth"))
    parser.add_argument("--output-model", type=Path, default=Path("model/v1/output/classifier_usrp_magphase16_real.ckpt"))
    parser.add_argument("--metrics-json", type=Path, default=Path("model/v1/output/classifier_usrp_magphase16_real_metrics.json"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload = torch.load(args.dataset)
    features = payload.get("features_flat", payload.get("rssi")).float()
    labels = payload["label"].long()
    metadata = payload.get("metadata", {})

    input_dim = int(features.shape[1])
    labels_np = labels.numpy()
    present_labels = sorted(np.unique(labels_np).astype(int).tolist())
    num_classes = max(present_labels) + 1

    train_idx, valid_idx, test_idx = stratified_split(labels_np, args.valid_frac, args.test_frac, args.seed)
    print(f"Dataset: {args.dataset}")
    print(f"features: {tuple(features.shape)}, labels: {len(present_labels)} present classes")
    print(f"present labels: {present_labels}")
    print(f"missing labels: {metadata.get('missing_labels', [])}")
    print(f"split: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    train_loader = make_loader(features, labels, train_idx, args.batch_size, True)
    valid_loader = make_loader(features, labels, valid_idx, args.batch_size, False)
    test_loader = make_loader(features, labels, test_idx, args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocationClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=20, factor=0.5)

    best_valid_acc = -1.0
    best_epoch = 0
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        valid_acc, _, _ = evaluate(model, valid_loader, device)
        scheduler.step(valid_acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_model)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch:03d}/{args.epochs}: loss={avg_loss:.4f}, valid_acc={valid_acc:.4f}")

    model.load_state_dict(torch.load(args.output_model, map_location=device))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    test_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    test_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    knn_metrics = run_knn(features.numpy(), labels_np, train_idx, test_idx, args.knn_k)

    print(f"Best valid accuracy: {best_valid_acc:.4f} @ epoch {best_epoch}")
    print(f"MLP Test Accuracy: {test_acc:.4f}")
    print(f"MLP Test Recall: {test_recall:.4f}")
    print(f"MLP Test Precision: {test_precision:.4f}")
    print(report)
    print(f"KNN(k={args.knn_k}) Test Accuracy: {knn_metrics['accuracy']:.4f}")

    metrics = {
        "dataset": str(args.dataset),
        "input_dim": input_dim,
        "num_classes_output": num_classes,
        "present_labels": present_labels,
        "missing_labels": metadata.get("missing_labels", []),
        "label_counts": metadata.get("label_counts", {}),
        "split": {
            "train": int(len(train_idx)),
            "valid": int(len(valid_idx)),
            "test": int(len(test_idx)),
        },
        "best_valid_accuracy": float(best_valid_acc),
        "best_epoch": int(best_epoch),
        "mlp": {
            "accuracy": float(test_acc),
            "recall": float(test_recall),
            "precision": float(test_precision),
            "classification_report": report,
        },
        "knn": knn_metrics,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved model   -> {args.output_model}")
    print(f"Saved metrics -> {args.metrics_json}")


if __name__ == "__main__":
    main()

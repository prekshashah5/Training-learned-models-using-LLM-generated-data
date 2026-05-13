import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import DataLoader, Subset

from mscn.util import *
from mscn.data import get_train_datasets
from mscn.model import SetConv

DEVICE = torch.device("cpu")


# ============================================================
# Setup results directory
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join("results", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

# ============================================================
# Utilities
# ============================================================

def compute_qerrors(preds_norm, labels_norm, min_val, max_val):
    preds = unnormalize_labels(preds_norm, min_val, max_val).astype(np.float64)
    labels = unnormalize_labels(labels_norm, min_val, max_val).astype(np.float64)

    # Avoid division by zero
    preds = np.maximum(preds, 1e-10)
    labels = np.maximum(labels, 1e-10)

    qerrors = np.maximum(preds / labels, labels / preds)
    return qerrors


def qerror_loss(preds, targets, min_val, max_val):
    preds = (preds * (max_val - min_val)) + min_val
    targets = (targets * (max_val - min_val)) + min_val

    preds = torch.exp(preds)
    targets = torch.exp(targets)

    return torch.mean(torch.max(preds / targets,
                                targets / preds))



def plot_predicted_vs_actual(preds_norm,
                             labels_norm,
                             min_val,
                             max_val,
                             title,
                             filename):

    preds = unnormalize_labels(preds_norm, min_val, max_val)
    labels = unnormalize_labels(labels_norm, min_val, max_val)

    preds = np.array(preds)
    labels = np.array(labels)

    plt.figure(figsize=(6, 6))
    plt.scatter(labels, preds, alpha=0.4)

    max_plot = max(labels.max(), preds.max())

    plt.plot([1, max_plot],
             [1, max_plot],
             linestyle="--",
             color="red",
             label="Ideal")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True Cardinality")
    plt.ylabel("Predicted Cardinality")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


# ============================================================
# Train for N epochs (continue training)
# ============================================================

def train_epochs(model,
                 dataset, min_val, max_val,
                 sample_feats,
                 predicate_feats,
                 join_feats,
                 batch_size,
                 epochs,
                 cuda):

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            samples, predicates, joins, targets, \
            sample_masks, predicate_masks, join_masks = batch

            if DEVICE.type == "cuda":
                samples, predicates, joins, targets = samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE), targets.to(DEVICE)
                sample_masks, predicate_masks, join_masks = sample_masks.to(DEVICE), predicate_masks.to(DEVICE), join_masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(samples,
                            predicates,
                            joins,
                            sample_masks,
                            predicate_masks,
                            join_masks)

            loss = qerror_loss(outputs, targets, min_val, max_val)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch loss: {total_loss/len(loader)}")


def evaluate(model, dataset, batch_size, cuda):
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    preds = []
    labels = []

    for batch in loader:
        samples, predicates, joins, targets, \
        sample_masks, predicate_masks, join_masks = batch

        if DEVICE.type == "cuda":
            samples, predicates, joins, targets = samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE), targets.to(DEVICE)
            sample_masks, predicate_masks, join_masks = sample_masks.to(DEVICE), predicate_masks.to(DEVICE), join_masks.to(DEVICE)

        with torch.no_grad():
            outputs = model(samples,
                            predicates,
                            joins,
                            sample_masks,
                            predicate_masks,
                            join_masks)

        preds.extend(outputs.cpu().numpy().flatten())
        labels.extend(targets.cpu().numpy().flatten())

    return preds, labels


# ============================================================
# Learning Strategy (Random or Active)
# ============================================================

def run_strategy(full_dataset,
                 val_dataset,
                 sample_feats,
                 predicate_feats,
                 join_feats,
                 hid_units,
                 min_val,
                 max_val,
                 batch_size,
                 epochs_per_round,
                 rounds,
                 acquire,
                 cuda,
                 strategy="random"):

    n_total = len(full_dataset)
    n_initial = int(0.2 * n_total)

    indices = np.random.permutation(n_total)
    labeled_idx = list(indices[:n_initial])
    pool_idx = list(indices[n_initial:])

    model = SetConv(sample_feats,
                    predicate_feats,
                    join_feats,
                    hid_units)

    if DEVICE.type == "cuda":
        model.to(DEVICE)

    labeled_sizes = []
    median_errors = []

    for r in range(rounds):

        print(f"\n{strategy.upper()} ROUND {r}")
        print("Labeled:", len(labeled_idx))

        labeled_dataset = Subset(full_dataset, labeled_idx)

        # Continue training same model
        train_epochs(model,
                     labeled_dataset, min_val, max_val,
                     sample_feats,
                     predicate_feats,
                     join_feats,
                     batch_size,
                     epochs_per_round,
                     cuda=DEVICE.type == "cuda")

        preds_val, labels_val = evaluate(model,
                                         val_dataset,
                                         batch_size,
                                         cuda=DEVICE.type == "cuda")

        qerrors = compute_qerrors(preds_val,
                                  labels_val,
                                  min_val,
                                  max_val)

        median_q = np.median(qerrors)
        print("Validation Median Q-error:", median_q)

        labeled_sizes.append(len(labeled_idx))
        median_errors.append(median_q)

        if len(pool_idx) == 0:
            break

        acquire_now = min(acquire, len(pool_idx))

        if strategy == "random":
            new_indices = np.random.choice(pool_idx,
                                           acquire_now,
                                           replace=False)

        else:
            pool_subset = Subset(full_dataset, pool_idx)
            preds_pool, _ = evaluate(model,
                                     pool_subset,
                                     batch_size,
                                     cuda=DEVICE.type == "cuda")

            preds_pool = np.array(preds_pool)
            uncertainties = np.abs(preds_pool - np.mean(preds_pool))

            sorted_idx = np.argsort(uncertainties)[-acquire_now:]
            new_indices = [pool_idx[i] for i in sorted_idx]

        labeled_idx.extend(new_indices)
        pool_idx = list(set(pool_idx) - set(new_indices))

    return model, labeled_sizes, median_errors


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--env", action="store_true", help="load argument defaults from .env file")
    args = parser.parse_args()

    # If --env is passed, override defaults with .env values
    if args.env:
        from dotenv import load_dotenv
        load_dotenv()

        _defaults = parser.parse_args([args.testset])
        if args.queries == _defaults.queries:
            args.queries = int(os.getenv("TOTAL_QUERIES", args.queries))
        if args.epochs == _defaults.epochs:
            args.epochs = int(os.getenv("AL_EPOCHS", args.epochs))
        if args.rounds == _defaults.rounds:
            args.rounds = int(os.getenv("AL_ROUNDS", args.rounds))
        if args.batch == _defaults.batch:
            args.batch = int(os.getenv("BATCH_SIZE", args.batch))
        if args.hid == _defaults.hid:
            args.hid = int(os.getenv("HIDDEN_UNITS", args.hid))

        print(f"[env] Loaded defaults from .env: queries={args.queries}, epochs={args.epochs}, "
              f"rounds={args.rounds}, batch={args.batch}, hid={args.hid}")
    
    global DEVICE
    DEVICE = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    num_materialized_samples = 1000

    dicts, column_min_max_vals, min_val, max_val, \
    labels_train, labels_test, max_num_joins, max_num_predicates, \
    train_data, test_data = get_train_datasets(
        args.queries,
        num_materialized_samples
    )

    table2vec, column2vec, op2vec, join2vec = dicts

    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    # ---------------- FULL SUPERVISION ----------------

    print("\nFULL SUPERVISION")

    full_model = SetConv(sample_feats,
                         predicate_feats,
                         join_feats,
                         args.hid)

    if DEVICE.type == "cuda":
        full_model.to(DEVICE)

    train_epochs(full_model,
                 train_data,
                    min_val, max_val,
                 sample_feats,
                 predicate_feats,
                 join_feats,
                 args.batch,
                 args.epochs * args.rounds,
                 cuda=DEVICE.type == "cuda")

    preds_full, labels_full = evaluate(full_model,
                                       test_data,
                                       args.batch,
                                       cuda=DEVICE.type == "cuda")

    plot_predicted_vs_actual(preds_full,
                             labels_full,
                             min_val,
                             max_val,
                             "Full Supervision",
                             "full_supervision.png")

    full_q = np.median(compute_qerrors(preds_full,
                                       labels_full,
                                       min_val,
                                       max_val))

    # ---------------- RANDOM ----------------

    random_model, random_sizes, random_errors = run_strategy(
        train_data,
        test_data,
        sample_feats,
        predicate_feats,
        join_feats,
        args.hid,
        min_val,
        max_val,
        args.batch,
        args.epochs,
        args.rounds,
        500,
        cuda=DEVICE.type == "cuda",
        strategy="random"
    )

    preds_random, labels_random = evaluate(random_model,
                                           test_data,
                                           args.batch,
                                           cuda=DEVICE.type == "cuda")

    plot_predicted_vs_actual(preds_random,
                             labels_random,
                             min_val,
                             max_val,
                             "Random Sampling",
                             "random_sampling.png")

    # ---------------- ACTIVE LEARNING ----------------

    al_model, al_sizes, al_errors = run_strategy(
        train_data,
        test_data,
        sample_feats,
        predicate_feats,
        join_feats,
        args.hid,
        min_val,
        max_val,
        args.batch,
        args.epochs,
        args.rounds,
        500,
        cuda=DEVICE.type == "cuda",
        strategy="uncertainty"
    )

    preds_al, labels_al = evaluate(al_model,
                                   test_data,
                                   args.batch,
                                   cuda=DEVICE.type == "cuda")

    plot_predicted_vs_actual(preds_al,
                             labels_al,
                             min_val,
                             max_val,
                             "Active Learning",
                             "active_learning.png")

    # ---------------- LEARNING CURVE ----------------

    plt.figure()
    plt.plot(random_sizes, random_errors,
             marker="o", label="Random")

    plt.plot(al_sizes, al_errors,
             marker="o", label="Active Learning")

    plt.axhline(full_q,
                linestyle="--",
                label="Full Supervision")

    plt.xlabel("Number of Labeled Queries")
    plt.ylabel("Validation Median Q-error")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR,
                             "learning_curve.png"))

    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()


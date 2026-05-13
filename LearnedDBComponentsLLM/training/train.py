import argparse
import time
import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset
from mscn.model import SetConv

import matplotlib.pyplot as plt
from datetime import datetime

# Global configuration
RESULTS_DIR = ""
DEVICE = torch.device("cpu")

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def set_seed(seed):
    if seed is object:
        seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qerror_loss(preds, targets, min_val, max_val):
    preds = unnormalize_torch(preds, min_val, max_val).flatten()
    targets = unnormalize_torch(targets, min_val, max_val).flatten()
    qerror = torch.max(preds / targets, targets / preds)
    return torch.mean(qerror)


def predict(model, data_loader, cuda):
    preds = []
    actuals = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if DEVICE.type == "cuda":
            samples, predicates, joins, targets = samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE), targets.to(DEVICE)
            sample_masks, predicate_masks, join_masks = sample_masks.to(DEVICE), predicate_masks.to(DEVICE), join_masks.to(DEVICE)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.shape[0]):
            preds.append(outputs[i].item())
            actuals.append(targets[i].item())

    return preds, actuals, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    labels_unnorm = np.array(labels_unnorm, dtype=np.float64).flatten()
    preds_unnorm = np.array(preds_unnorm, dtype=np.float64).flatten()
    
    # Avoid division by zero
    preds_unnorm = np.maximum(preds_unnorm, 1e-10)
    labels_unnorm = np.maximum(labels_unnorm, 1e-10)
    
    qerror = np.maximum(labels_unnorm / preds_unnorm, preds_unnorm / labels_unnorm)

    print(f"Median: {np.median(qerror)}")
    print(f"90th percentile: {np.percentile(qerror, 90)}")
    print(f"95th percentile: {np.percentile(qerror, 95)}")
    print(f"99th percentile: {np.percentile(qerror, 99)}")
    print(f"Max: {np.max(qerror)}")
    print(f"Mean: {np.mean(qerror)}")

def plot_predicted_vs_actual(preds_norm, labels_norm, min_val, max_val, title, filename):

    # Convert tensors to floats
    preds_norm = [p.item() if hasattr(p, "item") else float(p) for p in preds_norm]
    labels_norm = [l.item() if hasattr(l, "item") else float(l) for l in labels_norm]

    preds = unnormalize_labels(preds_norm, min_val, max_val)
    true_vals = unnormalize_labels(labels_norm, min_val, max_val)

    preds = np.array(preds)
    true_vals = np.array(true_vals)

    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, preds, alpha=0.4)

    max_plot = max(true_vals.max(), preds.max())

    plt.plot([1, max_plot], [1, max_plot],
             linestyle="--", color="red", label="Ideal")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("True Cardinality")
    plt.ylabel("Predicted Cardinality")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, min_val, max_val, epochs, cuda, optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(epochs):
        loss_total = 0.

        for data_batch in train_loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if DEVICE.type == "cuda":
                samples, predicates, joins, targets = samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE), targets.to(DEVICE)
                sample_masks, predicate_masks, join_masks = sample_masks.to(DEVICE), predicate_masks.to(DEVICE), join_masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(samples, predicates, joins,
                            sample_masks, predicate_masks, join_masks)

            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        print(f"Epoch {epoch}, loss: {loss_total / len(train_loader)}")

def active_learning_loop(
        full_dataset,
        val_dataset,
        sample_feats,
        predicate_feats,
        join_feats,
        hid_units,
        min_val,
        max_val,
        batch_size,
        cuda,
        rounds=5,
        init_frac=0.2,
        acquire=500,
        epochs=10,
        strategy="random"):

    print("\n===== ACTIVE LEARNING START =====")
    
    n_total = len(full_dataset)
    n_initial = int(acquire)
    print("full len: ", n_total)
    print("initial len: ", n_initial)
    indices = np.random.permutation(n_total)
    labeled_idx = list(indices[:n_initial])
    pool_idx = list(indices[n_initial:])

    labeled_sizes = []
    median_errors = []

    # Initialize model inside the loop to train from scratch each round
    model = SetConv(sample_feats,
                    predicate_feats,
                    join_feats,
                    hid_units)

    if DEVICE.type == "cuda":
        model.to(DEVICE)

    for r in range(rounds):

        print(f"\n===== Round {r} =====")
        print("Labeled:", len(labeled_idx), "Pool:", len(pool_idx))

        labeled_dataset = Subset(full_dataset, labeled_idx)

        train_loader = DataLoader(labeled_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size)

        train_model(model, train_loader,
                    min_val, max_val,
                    epochs, cuda=DEVICE.type == "cuda")

        # ---- TRAINING PLOT (AL round) ----
        preds_train, labels_train_aligned, _ = predict(model, train_loader, cuda=DEVICE.type == "cuda")

        plot_predicted_vs_actual(
            preds_train,
            labels_train_aligned,
            min_val,
            max_val,
            title=f"AL Round {r} - TRAIN",
            filename=f"al_round_{r}_train.png"
        )

        # ---- VALIDATION PLOT (AL round) ----
        preds_val, labels_val_aligned, _ = predict(model, val_loader, cuda=DEVICE.type == "cuda")

        plot_predicted_vs_actual(
            preds_val,
            labels_val_aligned,
            min_val,
            max_val,
            title=f"AL Round {r} - VALIDATION",
            filename=f"al_round_{r}_val.png"
        )

        # ---- Compute validation median Q-error ----
        preds_val_unnorm = unnormalize_labels(preds_val,
                                              min_val,
                                              max_val)

        val_labels_unnorm = unnormalize_labels(labels_val_aligned,
                                               min_val,
                                               max_val)

        qerrors = []
        for p, t in zip(preds_val_unnorm, val_labels_unnorm):
            p = float(p)
            t = float(t)
            qerrors.append(max(p/t, t/p))

        median_q = np.median(qerrors)
        print("Validation Median Q-error:", median_q)

        labeled_sizes.append(len(labeled_idx))
        median_errors.append(median_q)

        # ---- Acquisition ----
        if len(pool_idx) == 0:
            break

        acquire_now = min(acquire, len(pool_idx))
        
        if strategy == "uncertainty":
            print(f"Acquiring {acquire_now} samples using uncertainty sampling (Q-error)...")
            pool_dataset = Subset(full_dataset, pool_idx)
            pool_loader = DataLoader(pool_dataset, batch_size=batch_size)
            
            # Predict on pool to find high-error samples
            preds_pool, labels_pool_aligned, _ = predict(model, pool_loader, cuda=DEVICE.type == "cuda")
            
            # Unnormalize to compute real Q-error
            preds_pool_unnorm = unnormalize_labels(preds_pool, min_val, max_val)
            labels_pool_unnorm = unnormalize_labels(labels_pool_aligned, min_val, max_val)
            
            pool_qerrors = []
            for p, t in zip(preds_pool_unnorm, labels_pool_unnorm):
                p, t = float(p), float(t)
                pool_qerrors.append(max(p/t, t/p))
            
            # Sort pool indices by Q-error descending
            sorted_pool_rel_idx = np.argsort(pool_qerrors)[::-1]
            new_rel_indices = sorted_pool_rel_idx[:acquire_now]
            new_indices = [pool_idx[i] for i in new_rel_indices]

        elif strategy == "mc_dropout":
            num_samples_mc = 25
            print(f"Acquiring {acquire_now} samples using MC Dropout sampling (T={num_samples_mc})...")
            pool_dataset = Subset(full_dataset, pool_idx)
            pool_loader = DataLoader(pool_dataset, batch_size=batch_size)
            
            model.train() # Keep dropout active
            all_mc_preds = []
            
            for _ in range(num_samples_mc):
                mc_preds = []
                for batch in pool_loader:
                    samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = batch
                    if DEVICE.type == "cuda":
                        samples, predicates, joins = samples.to(DEVICE), predicates.to(DEVICE), joins.to(DEVICE)
                        sample_masks, predicate_masks, join_masks = sample_masks.to(DEVICE), predicate_masks.to(DEVICE), join_masks.to(DEVICE)
                    
                    outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
                    mc_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_mc_preds.append(mc_preds)
            
            all_mc_preds = np.array(all_mc_preds) # Shape: [T, PoolSize]
            pool_variances = np.var(all_mc_preds, axis=0) # Variance across T samples
            
            # Sort by variance descending
            sorted_pool_rel_idx = np.argsort(pool_variances)[::-1]
            new_rel_indices = sorted_pool_rel_idx[:acquire_now]
            new_indices = [pool_idx[i] for i in new_rel_indices]
            
        else:
            # Random acquisition
            new_indices = np.random.choice(pool_idx,
                                           acquire_now,
                                           replace=False)

        labeled_idx.extend(new_indices)
        pool_idx = list(set(pool_idx) - set(new_indices))

    return labeled_sizes, median_errors

def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda, strategy, rounds, out_dir, acquire, sup_queries=None):
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR = os.path.join(out_dir, timestamp)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # If sup-queries is specified and we are in supervised mode, override num_queries
    actual_num_queries = num_queries
    if strategy == "supervised" and sup_queries is not None:
        actual_num_queries = sup_queries
        print(f"Supervised model will train on custom budget: {actual_num_queries} queries")

    # Load training and validation data
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        actual_num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    DEVICE = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if DEVICE.type == "cuda":
        model.to(DEVICE)

    # Load test data
    file_name = "workloads/" + workload_name
    print(file_name)
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, labels_test_aligned, t_total = predict(model, test_data_loader, cuda=DEVICE.type == "cuda")
    print("Prediction time per test sample: {}".format(t_total / len(labels_test_aligned) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    print_qerror(preds_test_unnorm, label)

    # Write predictions
    file_name = "results/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")

    if strategy == "supervised":
        print("\n===== FULL SUPERVISED TRAINING (Epoch-by-Epoch Progress) =====")
        full_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        # Use total budget: rounds * num_epochs
        total_epochs = num_epochs * rounds
        
        labeled_sizes = []
        median_errors = []
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for e in range(total_epochs):
            # Train for 1 epoch
            train_model(model, full_train_loader, min_val, max_val, 1, cuda=DEVICE.type == "cuda", optimizer=optimizer)
            
            # Evaluate after each epoch
            preds_val, labels_val_aligned, _ = predict(model, test_data_loader, cuda=DEVICE.type == "cuda")
            preds_val_unnorm = unnormalize_labels(preds_val, min_val, max_val)
            val_labels_unnorm = unnormalize_labels(labels_val_aligned, min_val, max_val)
            
            # Vectorized qerror
            preds_val_unnorm = np.array(preds_val_unnorm, dtype=np.float64).flatten()
            val_labels_unnorm = np.array(val_labels_unnorm, dtype=np.float64).flatten()
            preds_val_unnorm = np.maximum(preds_val_unnorm, 1e-10)
            val_labels_unnorm = np.maximum(val_labels_unnorm, 1e-10)
            qerrors = np.maximum(preds_val_unnorm / val_labels_unnorm, val_labels_unnorm / preds_val_unnorm)
            
            median_q = np.median(qerrors)
            print(f"Epoch {e+1}/{total_epochs} - Validation Median Q-error: {median_q}")
            
            labeled_sizes.append(len(train_data))
            median_errors.append(median_q)
    else:
        labeled_sizes, median_errors = active_learning_loop(
            full_dataset=train_data,
            val_dataset=test_data,
            sample_feats=sample_feats,
            predicate_feats=predicate_feats,
            join_feats=join_feats,
            hid_units=hid_units,
            min_val=min_val,
            max_val=max_val,
            batch_size=batch_size,
            cuda=DEVICE.type == "cuda",
            rounds=rounds,
            init_frac=0.2,
            acquire=acquire,
            epochs=num_epochs,
            strategy=strategy
        )

    # Save learning curve data to CSV
    csv_path = os.path.join(RESULTS_DIR, "learning_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["labeled_size", "median_qerror", "cumulative_epochs"])
        for i, (sz, err) in enumerate(zip(labeled_sizes, median_errors)):
            if strategy == "supervised":
                cum_epochs = i + 1
            else:
                # For AL, we evaluate after training on the acquired set for num_epochs
                cum_epochs = (i + 1) * num_epochs
            writer.writerow([sz, err, cum_epochs])

    plt.figure()
    if strategy == "supervised":
        plt.plot(np.arange(1, len(median_errors) + 1), median_errors, marker='o')
        plt.xlabel("Epochs")
    else:
        plt.plot(labeled_sizes, median_errors, marker='o')
        plt.xlabel("Number of Labeled Samples")
    
    plt.yscale('log')
    plt.ylabel("Validation Median Q-error")
    plt.title(f"{strategy.capitalize()} Acquisition Learning Curve")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(RESULTS_DIR, f"learning_curve_{strategy}.png"))
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--strategy", help="random, uncertainty, mc_dropout, or supervised (default: random)", type=str, default="random")
    parser.add_argument("--rounds", help="number of active learning rounds (default: 5)", type=int, default=5)
    parser.add_argument("--out", help="base directory for results (default: results)", type=str, default="results")
    parser.add_argument("--acquire", help="acquisition size per round (default: 500)", type=int, default=500)
    parser.add_argument("--seed", help="random seed", type=int, default=None)
    parser.add_argument("--sup-queries", help="number of queries for supervised strategy (if different from --queries)", type=int, default=None)
    parser.add_argument("--env", help="load argument defaults from .env file", action="store_true")
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
        if args.batch == _defaults.batch:
            args.batch = int(os.getenv("BATCH_SIZE", args.batch))
        if args.hid == _defaults.hid:
            args.hid = int(os.getenv("HIDDEN_UNITS", args.hid))
        if args.rounds == _defaults.rounds:
            args.rounds = int(os.getenv("AL_ROUNDS", args.rounds))
        if args.acquire == _defaults.acquire:
            args.acquire = int(os.getenv("AL_ACQUIRE", args.acquire))

        print(f"[env] Loaded defaults from .env: queries={args.queries}, epochs={args.epochs}, "
              f"batch={args.batch}, hid={args.hid}, rounds={args.rounds}, acquire={args.acquire}")

    # Set seed for reproducibility
    if args.seed:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    train_and_predict(args.testset, args.queries, args.epochs, args.batch, args.hid, args.cuda, args.strategy, args.rounds, args.out, args.acquire, args.sup_queries)

if __name__ == "__main__":
    main()


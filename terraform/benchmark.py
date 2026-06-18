import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print("=" * 60)
print("LightGBM Credit Card Fraud Detection Benchmark")
print("Instance: r5.2xlarge (8 vCPU, 32 GB RAM)")
print("=" * 60)

# 1. Load data
print("\n[1/5] Loading dataset...")
t0 = time.time()
df = pd.read_csv("creditcard.csv")
load_time = time.time() - t0
print(f"  Rows: {len(df):,} | Columns: {len(df.columns)} | Load time: {load_time:.2f}s")

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Fraud rate: {y.mean()*100:.3f}%")

# 2. Train
print("\n[2/5] Training LightGBM...")
params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "is_unbalance": True,
    "verbosity": -1,
    "n_jobs": -1,
}
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

t1 = time.time()
callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=100)]
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=callbacks,
)
train_time = time.time() - t1
print(f"  Training time: {train_time:.2f}s | Best iteration: {model.best_iteration}")

# 3. Evaluate
print("\n[3/5] Evaluating...")
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"  AUC-ROC:   {auc:.6f}")
print(f"  Accuracy:  {acc:.6f}")
print(f"  F1-Score:  {f1:.6f}")
print(f"  Precision: {prec:.6f}")
print(f"  Recall:    {rec:.6f}")

# 4. Inference latency
print("\n[4/5] Inference latency...")
single_row = X_test.iloc[[0]]
runs = 100
t2 = time.time()
for _ in range(runs):
    model.predict(single_row)
latency_1row = (time.time() - t2) / runs * 1000  # ms

batch_1000 = X_test.iloc[:1000]
t3 = time.time()
model.predict(batch_1000)
throughput_1000 = (time.time() - t3) * 1000  # ms

print(f"  Inference latency (1 row):       {latency_1row:.3f} ms")
print(f"  Inference time (1000 rows batch): {throughput_1000:.3f} ms")

# 5. Save results
print("\n[5/5] Saving results...")
results = {
    "instance_type": "r5.2xlarge",
    "dataset": "creditcard.csv",
    "total_rows": int(len(df)),
    "load_time_s": round(load_time, 3),
    "train_time_s": round(train_time, 3),
    "best_iteration": int(model.best_iteration),
    "auc_roc": round(auc, 6),
    "accuracy": round(acc, 6),
    "f1_score": round(f1, 6),
    "precision": round(prec, 6),
    "recall": round(rec, 6),
    "inference_latency_1row_ms": round(latency_1row, 3),
    "inference_time_1000rows_ms": round(throughput_1000, 3),
}

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("  Saved: benchmark_result.json")
print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
for k, v in results.items():
    print(f"  {k}: {v}")

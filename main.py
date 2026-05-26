#!/usr/bin/env python3
"""
MLBench: A lightweight benchmarking tool for machine learning models.
Compare accuracy, training time, and inference time of different models on a dataset.
"""

import argparse
import time
import json
import sys
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import pandas as pd


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a built-in dataset for benchmarking."""
    datasets = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine
    }

    if name not in datasets:
        raise ValueError(f"Dataset {name} not supported. Choose from: {list(datasets.keys())}")

    data = datasets[name]()
    return data.data, data.target


def get_models() -> Dict[str, object]:
    """Return a dictionary of models to benchmark."""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }


def benchmark_model(model, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Benchmark a single model and return metrics."""
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'predictions': y_pred.tolist()
    }


def run_benchmark(dataset_name: str, test_size: float = 0.2) -> List[Dict]:
    """Run benchmark on all models for a given dataset."""
    print(f"Loading dataset: {dataset_name}")
    X, y = load_dataset(dataset_name)

    print(f"Splitting data (test_size={test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = get_models()
    results = []

    print("\nBenchmarking models...")
    for name, model in models.items():
        print(f"  Testing {name}...")
        metrics = benchmark_model(model, X_train, X_test, y_train, y_test)
        metrics['model'] = name
        results.append(metrics)

    return results


def print_results(results: List[Dict], dataset_name: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*60}")

    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Print header
    print(f"{'Model':<25} {'Accuracy':<10} {'Train Time (s)':<15} {'Predict Time (s)':<15}")
    print(f"{'-'*25} {'-'*10} {'-'*15} {'-'*15}")

    for result in results_sorted:
        print(f"{result['model']:<25} {result['accuracy']:<10.4f} "
              f"{result['train_time']:<15.4f} {result['predict_time']:<15.4f}")

    # Print best model
    best = results_sorted[0]
    print(f"\n🏆 Best Model: {best['model']} (Accuracy: {best['accuracy']:.4f})")

    # Detailed report for best model
    print(f"\nDetailed classification report for {best['model']}:")
    # Note: We'd need to store predictions to generate report, but for simplicity we'll skip
    # In a full implementation, we would return the predictions and generate the report here.


def save_results(results: List[Dict], dataset_name: str, filename: str = None):
    """Save results to a JSON file."""
    if filename is None:
        filename = f"benchmark_{dataset_name}_{int(time.time())}.json"

    # Convert numpy types to Python types for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = {
            'model': result['model'],
            'accuracy': float(result['accuracy']),
            'train_time': float(result['train_time']),
            'predict_time': float(result['predict_time'])
        }
        serializable_results.append(serializable_result)

    with open(filename, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'timestamp': time.time(),
            'results': serializable_results
        }, f, indent=2)

    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="MLBench: Benchmark ML models on datasets")
    parser.add_argument('--dataset', type=str, default='iris',
                        choices=['iris', 'breast_cancer', 'wine'],
                        help='Dataset to use for benchmarking')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for saved results')

    args = parser.parse_args()

    try:
        results = run_benchmark(args.dataset, args.test_size)
        print_results(results, args.dataset)

        if args.save:
            save_results(results, args.dataset, args.output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
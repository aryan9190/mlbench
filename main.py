#!/usr/bin/env python3
"""
MLBench: A lightweight benchmarking tool for machine learning models.
Compare accuracy, training time, and inference time of different models on a dataset.
"""

import argparse
import time
import json
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import pandas as pd


def load_dataset(name: str, csv_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset for benchmarking.
    If csv_path is provided, load from CSV (assuming last column is label).
    Otherwise, load built-in dataset by name.
    """
    if csv_path is not None:
        # Load custom dataset from CSV
        df = pd.read_csv(csv_path)
        # Assume last column is the label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    else:
        # Load built-in dataset
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


def run_benchmark(dataset_name: str, test_size: float = 0.2,
                  selected_models: Optional[List[str]] = None,
                  csv_path: Optional[str] = None,
                  cv_folds: int = 5) -> List[Dict]:
    """Run benchmark on selected models for a given dataset."""
    print(f"Loading dataset: {dataset_name}")
    X, y = load_dataset(dataset_name, csv_path)

    print(f"Splitting data (test_size={test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    all_models = get_models()

    # Filter models if specific ones were requested
    if selected_models is not None:
        models = {name: all_models[name] for name in selected_models if name in all_models}
        if not models:
            print(f"Warning: No valid models selected. Using all models.")
            models = all_models
    else:
        models = all_models

    results = []

    print(f"\nBenchmarking {len(models)} model(s)...")
    for name, model in models.items():
        print(f"  Testing {name}...")

        # Standard benchmark (train/test split)
        metrics = benchmark_model(model, X_train, X_test, y_train, y_test)
        metrics['model'] = name

        # Cross-validation score
        print(f"    Running {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        results.append(metrics)

    return results


def print_results(results: List[Dict], dataset_name: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*70}")

    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Print header
    print(f"{'Model':<25} {'Accuracy':<10} {'Train Time (s)':<15} {'Predict Time (s)':<15} {'CV Mean':<10} {'CV Std':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*15} {'-'*15} {'-'*10} {'-'*10}")

    for result in results_sorted:
        print(f"{result['model']:<25} {result['accuracy']:<10.4f} "
              f"{result['train_time']:<15.4f} {result['predict_time']:<15.4f} "
              f"{result.get('cv_mean', 0):<10.4f} {result.get('cv_std', 0):<10.4f}")

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
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file for custom dataset (last column should be labels)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['Random Forest', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree'],
                        help='Specific models to benchmark (default: all)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for saved results')

    args = parser.parse_args()

    try:
        results = run_benchmark(
            dataset_name=args.dataset,
            test_size=args.test_size,
            selected_models=args.models,
            csv_path=args.csv,
            cv_folds=args.cv
        )
        print_results(results, args.dataset)

        if args.save:
            save_results(results, args.dataset, args.output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
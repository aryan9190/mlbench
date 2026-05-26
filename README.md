# MLBench: Machine Learning Model Benchmarking Tool

A lightweight command-line tool for benchmarking and comparing machine learning models on standard datasets. MLBench helps data scientists and ML practitioners quickly evaluate model performance in terms of accuracy, training time, and inference time.

## Features

- Benchmark multiple ML models simultaneously
- Support for popular datasets (Iris, Breast Cancer, Wine) and custom CSV datasets
- Metrics: Accuracy, Training Time, Inference Time, Cross-Validation Scores
- Easy-to-read table output with CV mean and std
- JSON export for further analysis
- Minimal dependencies (only numpy, scikit-learn, pandas)
- Model selection via --models flag
- Configurable cross-validation folds

## Tech Stack

- **Language**: Python 3.7+
- **Libraries**: 
  - NumPy (numerical operations)
  - Scikit-learn (ML models and datasets)
  - Pandas (data handling)
- **Tools**: Standard Python packaging

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryan9190/mlbench.git
   cd mlbench
   ```

2. Install dependencies:
   ```bash
   pip install --break-system-packages -r requirements.txt
   ```

## Usage

Run the benchmark on the default Iris dataset:
```bash
python main.py
```

Specify a different dataset:
```bash
python main.py --dataset breast_cancer
```

Use a custom CSV dataset:
```bash
python main.py --csv path/to/your/data.csv
```

Adjust the test/train split:
```bash
python main.py --dataset wine --test-size 0.3
```

Benchmark specific models only:
```bash
python main.py --models "Random Forest" "SVM"
```

Adjust cross-validation folds:
```bash
python main.py --cv 3
```

Save results to a JSON file:
```bash
python main.py --save
```

Specify a custom output filename:
```bash
python main.py --save --output my_benchmark_results.json
```

## Example Output

```
Loading dataset: iris
Splitting data (test_size=0.2)

Benchmarking 5 model(s)...
  Testing Random Forest...
    Running 5-fold cross-validation...
  Testing SVM...
    Running 5-fold cross-validation...
  Testing Logistic Regression...
    Running 5-fold cross-validation...
  Testing K-Nearest Neighbors...
    Running 5-fold cross-validation...
  Testing Decision Tree...
    Running 5-fold cross-validation...

======================================================================
BENCHMARK RESULTS FOR IRIS
======================================================================
Model                     Accuracy   Train Time (s)  Predict Time (s) CV Mean    CV Std    
------------------------- ---------- --------------- --------------- ---------- ----------
K-Nearest Neighbors       1.0000     0.0006          0.0015          0.9733     0.0249    
SVM                       0.9667     0.0017          0.0002          0.9667     0.0211    
Logistic Regression       0.9667     0.0106          0.0002          0.9733     0.0249    
Decision Tree             0.9333     0.0009          0.0001          0.9533     0.0340    
Random Forest             0.9000     0.1097          0.0062          0.9667     0.0211    

🏆 Best Model: K-Nearest Neighbors (Accuracy: 1.0000)

Detailed classification report for K-Nearest Neighbors:

Results saved to benchmark_iris_1779786117.json
```

## Project Structure

```
mlbench/
├── main.py          # Main benchmarking script
├── requirements.txt # Python dependencies
├── README.md        # This file
└── .gitignore       # Git ignore file
```

## Customization

To add your own dataset:
1. Modify the `load_dataset` function in `main.py` to load your data
2. Ensure your data is in the format (X, y) where X is features and y is labels

To add more models:
1. Add your model to the `get_models` function in `main.py`
2. Ensure it follows the scikit-learn estimator interface

## License

MIT License - feel free to use and modify for your own projects.

## Acknowledgements

- Built with [Scikit-learn](https://scikit-learn.org/)
- Inspired by the need for quick model comparison in ML experiments
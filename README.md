# MLBench: Machine Learning Model Benchmarking Tool

A lightweight command-line tool for benchmarking and comparing machine learning models on standard datasets. MLBench helps data scientists and ML practitioners quickly evaluate model performance in terms of accuracy, training time, and inference time.

## Features

- Benchmark multiple ML models simultaneously
- Support for popular datasets (Iris, Breast Cancer, Wine)
- Metrics: Accuracy, Training Time, Inference Time
- Easy-to-read table output
- JSON export for further analysis
- Minimal dependencies (only numpy, scikit-learn, pandas)

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
   git clone https://github.com/yourusername/mlbench.git
   cd mlbench
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
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

Adjust the test/train split:
```bash
python main.py --dataset wine --test-size 0.3
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

Benchmarking models...
  Testing Random Forest...
  Testing SVM...
  Testing Logistic Regression...
  Testing K-Nearest Neighbors...
  Testing Decision Tree...

============================================================
BENCHMARK RESULTS FOR IRIS
============================================================
Model                      Accuracy    Train Time (s)   Predict Time (s)
-------------------------  ----------  -----------------  -----------------
Random Forest              0.9667      0.0123             0.0008
SVM                        0.9667      0.0045             0.0012
Logistic Regression        0.9667      0.0012             0.0005
K-Nearest Neighbors        0.9333      0.0001             0.0003
Decision Tree              0.9333      0.0008             0.0001

🏆 Best Model: Random Forest (Accuracy: 0.9667)

Results saved to benchmark_iris_1623456789.json
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
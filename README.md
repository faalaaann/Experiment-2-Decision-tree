# Experiment 2: Decision Tree Classification

This project trains and evaluates a decision tree classifier on the Wisconsin Breast Cancer dataset. The analysis is implemented in the Jupyter notebook [`EX 2 Decision tree.ipynb`](./EX%202%20Decision%20tree.ipynb), and the generated dataset is included in [`breast_cancer_dataset.csv`](./breast_cancer_dataset.csv).

## Objectives

- Load and verify the breast cancer dataset.
- Split the data into stratified training and testing sets.
- Train a pre-pruned decision tree classifier.
- Evaluate the model with accuracy, a classification report, and a confusion matrix.
- Visualize the resulting decision tree.

## Dataset

The notebook loads the scikit-learn breast cancer dataset, which contains:

- **569 samples**
- **30 numerical diagnostic features**
- **1 target column** (`Outcome`)
- **No missing values**

The target values represent malignant and benign diagnoses using the class labels provided by scikit-learn.

## Model configuration

The experiment uses `DecisionTreeClassifier` with:

```python
criterion="gini"
max_depth=4
min_samples_split=10
random_state=42
```

The data is split into 80% training data and 20% test data using stratification.

## Results

On the test set, the model achieved:

- **Accuracy:** 92.98%
- **Test samples:** 114
- **Confusion matrix:** `[[38, 4], [4, 68]]`

The notebook also prints precision, recall, and F1-score for both classes and saves a tree visualization as `decision_tree_structure.png` when it is executed.

## Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- scikit-learn

Install the dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Running the experiment

1. Clone this repository:

   ```bash
   git clone https://github.com/faalaaann/Experiment-2-Decision-tree.git
   cd Experiment-2-Decision-tree
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `EX 2 Decision tree.ipynb` and run all cells.

The notebook recreates `breast_cancer_dataset.csv`, evaluates the classifier, and generates the decision tree visualization.

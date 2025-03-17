# LDA vs QDA Classification: Comparison Analysis

## Dataset Overview

The analysis was performed on a dataset of approximately 46 transactions from a bankruptcy dataset. The data consisted of financial factors collected from bankrupt firms two years prior to their bankruptcy. The goal was to predict bankruptcy (binary outcome: 0 = non-bankrupt, 1 = bankrupt) based on the following four financial variables:

- **X1** = CF/TD = (cash flow) / (total debt)
- **X2** = NI/TA = (net income) / (total assets)
- **X3** = CA/CL = (current assets) / (current liabilities)
- **X4** = CA/NS = (current assets) / (net sales)

## Data Preparation

- The dataset was initially in `.xls` format and was converted into `.csv`.
- Headers were removed, and data was processed using **Pandas** and **NumPy** in Python.  
  (Refer to Figure 1 for the data handling process.)

## Techniques Selected

- **Linear Discriminant Analysis (LDA)**
- **Logistic (Quadratic) Discriminant Analysis (QDA)**

Two different sampling strategies were employed:
1. **Stratified K-Fold Sampling**
2. **Random Train-Test Sampling**

---

## Experiment 1: Train-Test Sampling on Both Classification Techniques

### Data Preparation

- The dataset was randomly split into training and testing sets using a 70:30 ratio.
- The **"stratify"** argument was used to ensure that both the training and testing sets maintained a proportional distribution of bankrupt (1) and non-bankrupt (0) firms. This ensured that the split had an equal representation of 0 and 1 values.
  
  (Refer to Figure 2 for the stratified train-test split and Figure 3 for the equal distribution of 0 and 1 data in the training and test samples.)

### Analysis and Results

After splitting the data, both LDA and QDA models were fitted to the training samples (X_train and Y_train) to predict bankruptcy. The accuracy results were as follows:

- **Linear Discriminant Analysis (LDA) Accuracy**: 83.3%
- **Logistic Discriminant Analysis (QDA) Accuracy**: 91.6%

#### Confusion Matrix

Confusion matrices were generated using heatmaps to assess model performance. The diagonals (True Positives and True Negatives) represent correct predictions, while the off-diagonals (False Positives and False Negatives) show incorrect predictions.

  (Refer to Figures 5 and 6 for confusion matrix heatmaps for LDA and QDA, respectively.)

---

## Experiment 2: Stratified K-Fold Sampling on Both Classification Techniques

### Data Preparation

- In **Stratified K-Fold Sampling**, the dataset was divided into **k = 10** folds while maintaining the same class proportions (0 and 1) in each fold.
- The **RepeatedStratifiedKFold** was used to split the data into k folds and repeat this process 3 times.

  (Refer to Figure 7 for the visualization of the stratified k-fold data splitting.)

### Analysis and Results

After splitting the data into folds, both LDA and QDA models were fitted to the training samples (X_train and Y_train) to predict bankruptcy. The accuracy results for both models were as follows:

- **Linear Discriminant Analysis (LDA) Accuracy**: 86.6%
- **Logistic Discriminant Analysis (QDA) Accuracy**: 87.8%

---

## Conclusion

- **Accuracy Comparison**:
  - In **Random Train-Test Sampling**, **QDA** outperformed **LDA** with an accuracy of 91.6% compared to 83.3% for LDA.
  - In **Stratified K-Fold Sampling**, **QDA** also had a slight edge over LDA, with an accuracy of 87.8% compared to 86.6%.

- **Sampling Strategy**:
  - The **Stratified K-Fold Sampling** approach provided a more robust estimate of model performance as it better handled the small dataset by ensuring the class distributions were preserved across folds.

Overall, **QDA** provided better results than **LDA**, likely due to its ability to model more complex decision boundaries (quadratic) compared to the linear decision boundaries of **LDA**. However, both models performed well and showed similar accuracy in **Stratified K-Fold Sampling**, where the model evaluations are more reliable due to repeated splits of the data.


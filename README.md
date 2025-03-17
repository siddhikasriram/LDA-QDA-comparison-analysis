# LDA-QDA-comparison-analysis
To determine which of the two classification algorithms, Linear Discriminant Analysis and Logistic Discriminant Analysis, when applied with two different sampling strategies, yields better results, and to provide reasoning for why it yields better results.

Dataset:
An analysis was carried out on a dataset of around 46 transactions obtained from a bankruptcy dataset. The annual financial data collected from the bankrupt firms two years prior to their bankruptcy. The following four factors were considered to predict if the bank was bankrupt or not.
The dataset has four key variables for prediction: X1 = CF/TD = (cash flow)/ (total debt)
X2 = NI/TA = (net income)/ (total assets)
X3 = CA/CL= (current assets)/ (current liabilities) X4 = CA/NS = (current assets)/ (net sales)
The deciding variable (Y) is binary: 0 for non-bankrupt and 1 for bankrupt. Data Preparation:
Data preparation involved converting the original .xls file into a CSV format. Headers were removed, and the data was processed using Python with Pandas and NumPy libraries. (Refer Figure 1)
Selection of techniques:
For this project, we selected two classification techniques: 1. Linear Discriminant Analysis (LDA)
2. Logistic (Quadratic) Discriminant Analysis (QDA)
We employed two different sampling strategies: 1. Stratified K-Fold Sampling
2. Random Train-Test Sampling
    
 Figure 1: Handling the headers
Experiment 1: Train-test sampling on both classification techniques Data preparation:
In the case of train-test sampling, the dataset was randomly split (70:30 ratio) into training and testing sets. To ensure that both the training and testing sets had a proportional representation of bankrupt and non-bankrupt firms, we utilized the "stratify" argument in the train-test split function. This stratification ensured an equal distribution of 0 and 1 values in both sets. (Refer figure 2.)
Figure 1: Using the stratify parameter
This parameter will stratify the split according to the 0 and 1 values. Here is the representation using counters to show how the data has been split among the training and the test samples. Refer figure 3.
    
 Figure 3: Equal splitting of 0 and 1 data among the training and test dataset
Analysis and Results:
After splitting the data, we fitted both the LDA and QDA models with the training samples (x train and y train) to predict bankruptcy. The accuracy results for both models were as follows (Refer figure 4):
• Linear Discriminant Analysis (LDA) Accuracy: 83.3%
• Logistic Discriminant Analysis (QDA) Accuracy: 91.6%
Figure 4: Accuracies of LDA and QDA after random sampling
Confusion Matrix:
We also created confusion matrices using heatmaps to assess the performance of these models. In a confusion matrix: (Refer figure 5 and 6)
• True Positives (TP) and True Negatives (TN) are on the diagonal (top left to bottom right), indicating correct predictions.
• False Positives (FP) and False Negatives (FN) are off diagonals, representing incorrect predictions.
 
 Figure 5: Confusion matrix heatmap for LDA
Figure 6: Confusion matrix heatmap for QDA
Experiment 2: Stratified K fold sampling on both classification techniques
In stratified k-fold sampling, the dataset is first divided into k equally sized (or nearly equally sized) subsets or folds while maintaining the proportional distribution of the target classes in each fold. This means that each fold will contain roughly the same ratio of 0s and 1s target variables as the original dataset. One of the folds is used as the validation set, while the remaining 10-1 folds are used as the training set (since k=10 in our case)
   
Data preparation for Stratified K fold sampling:
We simply make use of the RepeatedStratifiedKFold and split the data into k folds (k=10) and repeat the 10 folds, 3 times. Refer figure 7
Figure 7: Splitting the data into k folds using RepeatedStratifiedKFold
Analysis and Results:
After splitting the data, we fitted both the LDA and QDA modes with the training samples (x train and y train) to predict bankruptcy. The accuracy results for both models were as follows (Refer figure 8):
• Linear Discriminant Analysis (LDA) Accuracy: 86.6%
• Logistic Discriminant Analysis (QDA) Accuracy: 87.8%

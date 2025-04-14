import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score



true_labels = pd.read_csv('true_labels_debiased.csv')
predicted_labels = pd.read_csv('debias.csv')


assert all(true_labels.columns == predicted_labels.columns), "wrong column name"


best_thresholds = {}


for label in true_labels.columns:
    y_true = true_labels[label]
    y_pred = predicted_labels[label]
    

    best_f1 = -1
    best_threshold = 0.5  
    

    thresholds = np.linspace(0, 1, 100)  
    for threshold in thresholds:

        y_pred_binary = (y_pred >= threshold).astype(int)
        

        f1 = f1_score(y_true, y_pred_binary)
        

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    

    best_thresholds[label] = best_threshold


print("best threshold：", best_thresholds)


predicted_binary = predicted_labels.copy()
for label, threshold in best_thresholds.items():
    predicted_binary[label] = (predicted_binary[label] >= threshold).astype(int)


predicted_binary.to_csv('converted_pre_F1_debiased.csv', index=False)

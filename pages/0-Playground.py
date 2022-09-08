from locale import normalize
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, roc_curve, precision_recall_curve

st.set_page_config(layout="centered", page_title="Metrics 101")
sns.set_style("darkgrid")

# there are two columns in df_predictions: 'prediction' and 'ground_truth'
df_predictions = pd.read_csv("./data/predictions_logistic_regression.csv", index_col=0)


y_pred = df_predictions["prediction"]
y_true = df_predictions["ground_truth"]

# <--- side bar content --->
st.sidebar.markdown("# 1) Choose decision threshold")
# threshold slider
th = st.sidebar.slider(
    label="Decision Threshold", min_value=0.0, max_value=1.0, value=0.5
)
st.sidebar.markdown(
    """**Explanation**: for each sample in the dataset, the classifier outputs a probability.
To convert this probability to a class (0=customer stays, 1=customer leaves), we need a
decision threshold. A computed probability higher than the threshold translates to class 1,
while a probability lower than the threshold translates to class 0.
"""
)
st.sidebar.markdown("[Distribution of computed probabilities](#proba_dist)")
st.sidebar.markdown("[KPIs for current threshold](#current_kpis)")

# <--- enf of sidebar content --->

st.header("Distribution of computed probabilities", anchor="proba_dist")
"""
Below, the histogram plot for the probabilities output by the classifier is shown, 
colored using the ground-truth value. Overall, samples that
belong to class 1 tend to have a higher probability than those from class 0, which is good :-).
"""


# prediction hist plot
fig = plt.figure(figsize=(7, 3))
sns.histplot(
    data=df_predictions,
    x="prediction",
    hue="ground_truth",
    bins=30,
    palette={0: "#DCB9AF", 1: "#648D0D"},
    alpha=0.6
    # element="step",
)
y = [0, 60]
x = [th, th]

plt.plot(x, y, "k")
plt.xlim([0, 1])
plt.xlabel("Predicted probability")
st.pyplot(fig)


tp = len(df_predictions.query("prediction >= " + str(th) + " and ground_truth == 1"))
tn = len(df_predictions.query("prediction <" + str(th) + " and ground_truth == 0"))
fn = len(df_predictions.query("prediction < " + str(th) + " and ground_truth == 1"))
fp = len(df_predictions.query("prediction >=" + str(th) + " and ground_truth == 0"))
p = len(df_predictions.query("ground_truth == 1"))
n = len(df_predictions.query("ground_truth == 0"))
precision = tp / (tp + fp)
recall = tp / p

st.header("Performance for current threshold", anchor="current_kpis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Precision",
        "%.2f" % precision,
        help="ratio of relevant customers among selected ones",
    )

with col2:
    st.metric(
        "Recall",
        "%.2f" % recall,
        help="ratio of relevant customers caught",
    )

with col3:
    st.metric(
        "F1-score",
        "%.2f" % (2 * precision * recall / (precision + recall)),
        help="average of precision and recall",
    )
# with col1:
#     fig = plt.figure()
#     sns.kdeplot(
#         df_predictions.query("ground_truth == 1")["prediction"],
#         color="green",
#         label="positive class (churning)",
#         cut=0,
#     )
#     sns.kdeplot(
#         df_predictions.query("ground_truth == 0")["prediction"],
#         color="red",
#         label="negative class (staying)",
#         cut=0,
#     )
#     y = [0, 2]
#     x = [th, th]

#     plt.plot(x, y, "k")
#     plt.xlim([0, 1])
#     plt.xlabel("Predicted probability")
#     plt.title("Predicted probability distribution for both classes")
#     plt.legend()
#     st.pyplot(fig)


# precision-recall curve
"## Precision - Recall curve"
"""This curve is useful to find the optimal threshold to balance **precision**
(relevancy of retrieved items) versus **recall** (hit rate)"""

fig = plt.figure()
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
plt.plot(recall, precision, label="precision-recall")
threshold_id = np.argmin(abs(thresholds - th))
plt.plot(
    recall[threshold_id], precision[threshold_id], "ko", label="decision threshold"
)
plt.title("Precision - Recall Curve")
plt.xlabel("Recall (TP / P")
plt.ylabel("Precision (TP / (TP + FP)")
plt.legend()
st.pyplot(fig)

# F1-scores curve
"## F1 score curve"
"""F1-score is the mean of **precision** and **recall**, so it's relevant to find a 
compromise between the two. The curve shows the F1-score value for different thresholds"""
fig = plt.figure()
# https://stats.stackexchange.com/questions/518616/how-to-find-the-optimal-threshold-for-the-weighted-f1-score-in-a-binary-classifi
f1_scores = 2 * recall * precision / (recall + precision)
# precision and recal are size ((n_thresholds+1,) while thresholds is size (n_thresholds,)
# so we need to remove the last element of f1_score to have matching dimensions
plt.plot(thresholds, f1_scores[:-1], label="F1-scores")
plt.plot(
    thresholds[threshold_id],
    f1_scores[threshold_id],
    "ko",
    label="decision threshold",
)
plt.xlabel("Decision Threshold")
plt.ylabel("F1 score")
plt.title("F1-score curve")
plt.legend()
st.pyplot(fig)

# ROC-AUC curve
"## ROC curve"
"""ROC curve compares the **True Positive Rate** to the **False Positive Rate**. 
It's usually relevant for a balanced dataset, and when one focuses on the classifier
ability to distinguish between different classes
"""
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
threshold_id = np.argmin(abs(thresholds - th))
plt.plot(fpr, tpr, label="roc curve")
plt.plot(fpr[threshold_id], tpr[threshold_id], "ko", label="decision threshold")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve")
plt.legend()
st.pyplot(fig)

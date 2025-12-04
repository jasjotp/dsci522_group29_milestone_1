from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

# Training confusion matrix
disp_train = ConfusionMatrixDisplay.from_estimator(logreg, X_train, y_train)
disp_train.ax_.grid(False)
disp_train.ax_.set_title("Confusion Matrix of Logistic Regression using training data")
plt.show()

# Testing confusion matrix
disp_test = ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
disp_test.ax_.grid(False)
disp_test.ax_.set_title("Confusion Matrix of Logistic Regression using testing data")
plt.show()

# Micro-average AUC ROC (score)
y_score = logreg.predict_proba(X_test)
micro_roc_auc_ovr = roc_auc_score(
    y_test,
    y_score,
    multi_class="ovr",
    average="micro",
)

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

# Micro-averaged AUC ROC plot
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

display = RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange",
    plot_chance_level=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
)
plt.show()


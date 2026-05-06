import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Input path
#p = "predictions_hierarchical_median_448.npz"
p = "predictions_kmeans_median_448.npz"
data = np.load(p)
y_true = data["y_true"]
y_pred = data["y_pred"]

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["fake", "real"]
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - Kmeans Median") #title
plt.tight_layout()
#output
plt.savefig("confusion_matrix_kmeans_median_448.png")
print(cm)

from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class classifier:
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.
        """
        unique_classes = np.unique(y_true)  # Get unique class labels
        avg_mode = 'weighted' if len(unique_classes) > 2 else None
        print("Custom Accuracy")
        print(f"Accuracy: {accuracy_score(y_true,y_pred) *100:.2f}%")
        print(f"F1 score: {f1_score(y_true, y_pred,average=avg_mode)}")
        print(f"Recall: {recall_score(y_true, y_pred,average=avg_mode)}")
        print(f"Precision: {precision_score(y_true, y_pred,average=avg_mode)}")
        sns.heatmap(confusion_matrix(y_true,y_pred),annot=True,fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class cluster:
    def accuracy(self, y_pred):
        """
        Compute the accuracy of the model.
        """
        print(f"Silhouette: {silhouette_score(self.X,y_pred)}")
        print(f"Davies Bouldin: {davies_bouldin_score(self.X,y_pred)}")
        print(f"Calinski Harabasz: {calinski_harabasz_score(self.X,y_pred)}")
import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self,X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # calculate covariance
        cov = np.cov(X.T)

        # calculate eigenvector and eigenvalue
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self,X):
        # projects data
        X = X - self.mean
        return np.dot(X, self.components.T)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn import datasets

    data = datasets.load_wine()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X: ",X.shape)
    print("Shape of Projected X: ",X_projected.shape)

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    plt.scatter(x1,x2,c=y,edgecolors="none",alpha=0.8,cmap=matplotlib.colormaps.get_cmap('viridis'))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
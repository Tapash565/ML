from sklearn.preprocessing import PolynomialFeatures
from linear_regression import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

class PolynomialRegression(LinearRegression):
    def __init__(self, degree, n_iterations=500, learning_rate=0.01):
        self.degree = degree
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,learning_rate=learning_rate)
    
    def fit(self,X,y):
        poly = PolynomialFeatures(degree=self.degree)
        X = poly.fit_transform(X)
        super(PolynomialRegression, self).fit(X,y)
        
    
    def predict(self, X):
        poly = PolynomialFeatures(degree=self.degree)
        X = poly.fit_transform(X)
        return super(PolynomialRegression, self).predict(X)

    def accuracy(self,actual,predicted):
        r2 = r2_score(actual,predicted)
        mae = mean_absolute_error(actual,predicted)
        rmse = root_mean_squared_error(actual,predicted)
        print(f"R2 score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Square Error: {rmse}")

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X,y = make_regression(n_features=4,n_samples=2000,noise=1,random_state=12)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    model = PolynomialRegression(degree=3)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    model.accuracy(y_test,y_pred)
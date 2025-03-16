from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

class Regression:
    def accuracy(self,actual,predicted):
        r2 = r2_score(actual,predicted)
        mae = mean_absolute_error(actual,predicted)
        rmse = root_mean_squared_error(actual,predicted)
        print(f"R2 score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Square Error: {rmse}")
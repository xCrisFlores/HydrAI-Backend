from sklearn.linear_model import LinearRegression

class Regression:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
       
        self.model.fit(X, y)

    def predict(self, X):
      
        return self.model.predict(X)

    def score(self, X, y):
       
        return self.model.score(X, y)

    def get_coefficients(self):
       
        return self.model.coef_

    def get_intercept(self):
       
        return self.model.intercept_

class SaveableModel(object):
    def __init__(self, y_means, y_stdev, sk_model):
        self.y_means = y_means
        self.y_stdev = y_stdev
        self.sk_model = sk_model
    
    def rescale(self, y):
        return y*self.y_stdev + self.y_means

    def predict(self, X):
        y_pred = self.sk_model.predict(X)
        return self.rescale(y_pred)
    

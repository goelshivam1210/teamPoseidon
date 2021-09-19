import numpy as np

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
    
    @property
    def feature_importances_(self):
        return self.sk_model.feature_importances_
    
def dataset_to_arrays(dataset, flatten=True, last_y=True):
    all_xs = []
    all_ys = []
    for x, y in dataset.as_numpy_iterator():
        all_xs.append(x)
        all_ys.append(y)
    ys = np.concatenate(all_ys)
    xs = np.concatenate(all_xs)
    if flatten:
        xs = xs.reshape(xs.shape[0], -1)
    if last_y:
        ys = ys[:, -1]
    return xs, ys
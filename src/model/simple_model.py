from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier


class SimpleModel:
    def __init__(self, model_name, **kwargs):
        name = model_name.lower()
        if name == 'svm':
            self.model = SVC(**kwargs)
        elif name == 'rf':
            self.model = RandomForestClassifier(**kwargs)
        elif name == 'knn':
            self.model = KNeighborsClassifier(**kwargs)
        elif name == 'dt':
            self.model = DecisionTreeClassifier(**kwargs)
        elif name == 'ridge':
            self.model = RidgeClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown simple model: {name}")

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

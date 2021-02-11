class NaiveBayes:
    def __init__(self):
        self.data = None
        self.labels = None
        self.class_size = {}
        self.class_P = {}
        self.P = {}

    #-------------------------------------------------------
    def init_model(self, X_train, y_train):
        self.labels = y_train

        # concatenate X & y to easily get probabilities
        from pandas import concat
        self.data = concat([X_train, y_train], axis=1)

        for class_ in y_train.unique():
            # get each class size from labels(y)
            self.class_size[class_] = y_train.value_counts()[class_]

            # get each class probaility
            self.class_P[class_] = self.class_size[class_] / len(y_train.index)

        # initialize each feature dict
        for x in X_train.columns:
            self.P[x] = {}
            for class_ in y_train.unique():
                self.P[x][class_] = {}        

    #-------------------------------------------------------
    def train(self, X_train, y_train):
        self.init_model(X_train, y_train)

        # get conditional probability for each value 
        # in each feature(x), given each class in labels(y)
        # ex: P(X_train(x=val) | class)
        for x in X_train.columns:
            for val in X_train[x].unique():
                for class_ in y_train.unique():
                    # compute P(x_val | class_)
                    cond = self.data[(self.data[x] == val) & (self.data[y_train.name] == class_)]
                    cond_size = len(cond.index)
                    self.P[x][class_][val] = cond_size / self.class_size[class_]

    #-------------------------------------------------------
    def get_class_label(self, row):
        cond_P = {}  # conditional probability for each row given each class 
        for class_ in self.labels.unique():  
            P = []
            for x in row.index:
                P.append(self.P[x][class_][row[x]])  # append P(X(x=row[x]) | class_)
            P.append(self.class_P[class_])  # append probability of the class

            # multiplicatipn of each P(x_val | class_) and P(class_)
            from numpy import prod
            cond_P[class_] = prod(P)  
        
        max_class = max(cond_P, key=cond_P.get)
        row['class'] = max_class
        return row

    #-------------------------------------------------------
    def predict(self, X_test):
        X_test = X_test.apply(self.get_class_label, axis=1)
        return X_test['class']
        
    #-------------------------------------------------------
    def accuracy(self, df1, df2):
        total_size = len(df1.index)
        n_correct = (df1 == df2).sum()
        return n_correct / total_size


#-------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

# load the dataset
data = pd.read_csv('data/cars.csv')
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# load, train and test the model
model = NaiveBayes()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

model.train(X_train, y_train)
preds = model.predict(X_test)
print('Model accuracy:', model.accuracy(preds, y_test))


import sklearn as sk
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


# Load the data
ds = datasets.load_diabetes()

print(f'feature names:\n{ds.feature_names}')
print(ds.data[0], ds.target[0])

# fit the data
x = ds.data
y = ds.target

# split the data
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2)

# fit the model
clf = svm.SVR(kernel="linear", C=2)
clf.fit(x_train, y_train)

# evaluate
y_pred = clf.predict(x_test)
acc = metrics.r2_score(y_test, y_pred)

print(f'r2 score: {acc}')

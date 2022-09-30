import sklearn as sk
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt


# Load the data
ds = datasets.load_breast_cancer()

# visualize the data
print(ds.feature_names)

# fit the data
x = ds.data
y = ds.target

# split the data
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2)

# fit the model
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

# evaluate
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
# show the ROC curve
ax = plt.gca()
svc_disp = metrics.RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax)
svc_disp.plot(ax=ax)
plt.show()
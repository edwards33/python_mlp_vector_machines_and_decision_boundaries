from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.12, random_state=2)

model_SVC = SVC(kernel='poly')
model_SVC.fit(x_train, y_train)

predictions = model_SVC.predict(x_test)

print model_SVC.score(x_test, y_test)

print metrics.classification_report(y_test, predictions)
print metrics.confusion_matrix(y_test, predictions)

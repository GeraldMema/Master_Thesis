from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


dt1 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt2 = DecisionTreeClassifier(max_depth=5, random_state=42)

dt3 = DecisionTreeClassifier(max_depth=5, random_state=0)
dt4 = DecisionTreeClassifier(max_depth=5, random_state=0)

data, target = load_digits(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

m1 = dt1.fit(X_train, y_train)
m2 = dt2.fit(X_train, y_train)

m3 = dt3.fit(X_train, y_train)
m4 = dt4.fit(X_train, y_train)

p1 = m1.predict(X_test)
p2 = m2.predict(X_test)

p3 = m3.predict(X_test)
p4 = m4.predict(X_test)

print(f1_score(y_test, p1,average='micro'))
print(f1_score(y_test, p2,average='micro'))
print(f1_score(y_test, p3,average='micro'))
print(f1_score(y_test, p4,average='micro'))
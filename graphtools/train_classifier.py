from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_RandomForest(X_train, y_train, X_test):
    rf = RandomForestClassifier(max_depth=10, random_state=10)
    rf.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_score = [x[1] for x in rf.predict_proba(X_test)]

    return y_pred, y_score

def train_SVC(X_train, y_train, X_test):
    clf = SVC(gamma='auto', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = [x[1] for x in clf.predict_proba(X_test)]

    return y_pred, y_score
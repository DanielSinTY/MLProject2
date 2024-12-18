import numpy as np

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def clf_task(X_train,y_train,X_test,y_test):
    def evaluate_clf(clf, X_train, y_train, X_test, y_test):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("F1: ", f1_score(y_test, y_pred))

    print("Logistic Regression")
    clf = LogisticRegression(random_state=0)
    evaluate_clf(clf, X_train, y_train, X_test, y_test)
    print()

    print("Random Forest")
    clf = RandomForestClassifier(random_state=0)
    evaluate_clf(clf, X_train, y_train, X_test, y_test)
    feature_importances = clf.feature_importances_
    print("Feature importances: ", feature_importances)
    print()

    print("SVM")
    clf = SVC(random_state=0)
    evaluate_clf(clf, X_train, y_train, X_test, y_test)
    print()

    print("KNN")
    clf = KNeighborsClassifier()
    evaluate_clf(clf, X_train, y_train, X_test, y_test)
    print()

    print("MLP")
    clf = MLPClassifier(random_state=0)
    evaluate_clf(clf, X_train, y_train, X_test, y_test)

def reg_task( x_train, y_train, x_test, y_test, y_test_drug_binary):
    def evaluate_regression_model(model, X_train, y_train, X_test, y_test, y_test_drug_binary):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for group in "overall", "drug", "control":
            if group == "drug":
                mask = y_test_drug_binary
            elif group == "control":
                mask = ~y_test_drug_binary
            else:
                mask = np.ones(len(y_test), dtype=bool)
            mse = mean_squared_error(y_test[mask], y_pred[mask])
            mae = mean_absolute_error(y_test[mask], y_pred[mask])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[mask], y_pred[mask])
            print(f"{group.capitalize()} - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
    print("Linear Regression")
    model = LinearRegression()
    evaluate_regression_model(model, x_train, y_train, x_test, y_test, y_test_drug_binary)  
    print()

    print("SVR")
    model = SVR()
    evaluate_regression_model(model, x_train, y_train, x_test, y_test, y_test_drug_binary)
    print()

    print("Random Forest")
    model = RandomForestRegressor(random_state=0)
    evaluate_regression_model(model, x_train, y_train, x_test, y_test, y_test_drug_binary)
    feature_importances = model.feature_importances_
    print("Feature importances: ", feature_importances)
    print()

    print("Gaussian Process")
    kernel = RBF() + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    evaluate_regression_model(model, x_train, y_train, x_test, y_test, y_test_drug_binary)
    print()

    print("MLP")
    model = MLPRegressor(random_state=0)
    evaluate_regression_model(model, x_train, y_train, x_test, y_test, y_test_drug_binary)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_report, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge
import numpy as np

def train_model(df, model_name, task_type):
  target_col = df.columns[-1]
  X = df.drop(columns=[target_col])
  y = df[target_col]
  X = pd.get_dummies(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  model = None
  if task_type == "classification":
    if model_name == "Logistic Regression":
      model = LogisticRegression(max_iter = 100)
    elif model_name == "Random Forest Classifier":
      model = RandomForestClassifier()
    elif model_name == "SVM":
      model = SVC()
    elif model_name == "KNN":
      model = KNeighborsClassifier()
  elif task_type == "regression":
    if model_name == "Linear Regression":
      model = LinearRegression()
    elif model_name == "Random Forest Regressor":
      model = RandomForestRegressor()
    elif model_name == "SVM":
      model = SVR()
    elif model_name == "Ridge":
      model = Ridge()
  if model is None:
    return None, "Model not implemented."
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  if task_type == "classification":
    score = accuracy_score(y_test, predictions)
    return score, "Accuracy"
  elif task_type == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, "RMSE"

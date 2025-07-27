from sklearn.preprocessing import LabelEncoder

def detect_task_type(df):
  if df is None or df.empty:
    return "unknown"
  
  target_col = df.columns[-1]
  target = df[target_col]

  if target.dtype == 'object' or target.nunique() < 20:
    return "classification"
  elif target.dtype in ['int64', 'float64'] and target.nunique() >= 20:
    return "regression"
  else:
    return "unknown"

def suggest_models(task_type):
  if task_type == "classification":
    return ["Logistic Regression", "Random Forest Classifier", "SVM", "KNN"]
  elif task_type == "regression":
    return ["Linear Regression", "Random Forest Regressor", "SVR", "Ridge"]
  else:
    return []

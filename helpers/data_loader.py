import pandas as pd

def load_data(uploaded_file):
  file_type = uploaded_file.name.strip('.')[-1].lower()
  try:
    if file_type == 'csv':
      df = pd.read_csv(uploaded_file)
    elif file_type in ['xls', 'xlsx']:
      df = pd.read_excel(uploaded_file)
    elif file_type == 'json':
      df = pd.read_json(uploaded_file)
    else:
      return None, "Unsupported file format."
    return df, None
  except Exception as e:
    return None, str(e)

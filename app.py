from helpers.model_selector import detect_task_type, suggest_models
from helpers.data_loader import load_data
import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title = "ML Playground", layout = "wide")
st.title("🤖 Machine Learning Playground")

st.markdown("Upload your dataset and we'll help you explore, train models, or even invent new ones.")

# --- FIle Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel or JSON)", type = ['csv', 'xls', 'xlsx', 'json'])

if uploaded_file is not None:
  df, error = data_loader(uploaded_file)
  if error:
    st.error(f"⚠️ Error loading file: {error}")
  else:
    st.success(f"Successfully loaded your file!")
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("**Shape:**", df.shape)
    st.write("**Columns and Types:**")
    st.write(df.dtypes)

st.subheader("🧠 Task Type Detection")
task_type = detect_task_type(df)
st.write(f'Detected Task Type: `{task_type}`')

models = suggest_models(task_type)
if models:
  st.write("Suggested Model:")
  st.write(models)
else:
  st.warning("Couldn't suggest any models. Try another dataset or check the target column")

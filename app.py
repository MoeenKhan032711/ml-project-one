import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title = "ML Playground", layout = "wide")
st.title("🤖 Machine Learning Playground")

st.markdown("Upload your dataset and we'll help you explore, train models, or even invent new ones.")

# --- FIle Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel or JSON)", type = ['csv', 'xls', 'xlsx', 'json'])

if uploaded_file is not None:
  file_type = file_uploaded.name.split(".")[-1].lower()
  try:
    if file_type == 'csv':
      df = pd.read_csv(uploaded_file)
    elif file_type in ['xlsx', 'xls']:
      df = pd.read_excel(uploaded_file)
    elif filte_type == 'json':
      df = pd.read_json(uploaded_file)
    else:
      st.error("Unsupported file format.")
    if df is not None:
      st.success(f'Successfully loaded your **{file_type.upper()}** file!')
      st.subheader("📊 Data Preview")
      st.dataframe(df.head(), use_container_width = True)
      st.write("Shape:**", df.shape)
      st.write("Columns and Types:**")
      st.write(df.types)
  except Exception as e:
    st.error(f" ⚠️ Error loading file: {e}")

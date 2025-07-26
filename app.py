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

from helpers.model_selector import detect_task_type, suggest_models
from helpers.trainer import train_model
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

seleted_model = st.select_box("🎯 Choose one or more models to train", models)

if st.button("Train Model"):
  if not selected_models:
    st.warning("Please select at least one model.")
  else:
    results = []
    with st.spinner("Training models..."):
      for model_name in selected_models:
        score, metric_name = train_model(df, model_name, task_type)
        if score is not None:
          results.append((model_name, score))
        else:
          results.append((model_name, "❌ Error"))
  st.subheader("📈 Model Performance Comparison")
  st.table({"Model": [r[0] for r in results], metric_name: [r[1] for r in results]})

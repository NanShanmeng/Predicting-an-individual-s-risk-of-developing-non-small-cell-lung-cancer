import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
model = joblib.load('model.pkl')
X_test = pd.read_csv('X_test.csv')
feature_names = [
    "SLC6A13",
    "ANLN",
    "MARCO",
    "SYT13",
    "ARG2",
    "MEFV",
    "ZNF29P",
    "FLVCR2",
    "PTGFR",
    "CRISP2",
    "EME1",
    "IL22RA2",
    "SLC29A4",
    "CYBB",
    "LRRC25",
    "SCN8A",
    "LILRA6",
    "CTD_3080P12_3",
    "PECAM1"
]
st.title("Predicting the risk of non-small cell lung cancer based on the expression levels of diabetes-related genes")
SLC6A13 = st.number_input("SLC6A13", min_value=0, max_value=100000, value=161)
ANLN = st.number_input("ANLN", min_value=0, max_value=100000, value=161)
MARCO = st.number_input("MARCO", min_value=0, max_value=100000, value=1439)
SYT13 = st.number_input("SYT13", min_value=0, max_value=100000, value=12)
ARG2 = st.number_input("ARG2", min_value=0, max_value=100000, value=3224)
MEFV = st.number_input("MEFV", min_value=0, max_value=100000, value=34)
ZNF29P = st.number_input("ZNF29P", min_value=0, max_value=100000, value=1)
FLVCR2 = st.number_input("FLVCR2", min_value=0, max_value=100000, value=654)
PTGFR = st.number_input("PTGFR", min_value=0, max_value=100000, value=24)
CRISP2 = st.number_input("CRISP2", min_value=0, max_value=100000, value=44)
EME1 = st.number_input("EME1", min_value=0, max_value=100000, value=495)
IL22RA2 = st.number_input("IL22RA2", min_value=0, max_value=100000, value=12)
SLC29A4 = st.number_input("SLC29A4", min_value=0, max_value=100000, value=913)
CYBB = st.number_input("CYBB", min_value=0, max_value=100000, value=1629)
LRRC25 = st.number_input("LRRC25", min_value=0, max_value=100000, value=288)
SCN8A = st.number_input("SCN8A", min_value=0, max_value=100000, value=714)
LILRA6 = st.number_input("LILRA6", min_value=0, max_value=100000, value=113)
CTD_3080P12_3 = st.number_input("CTD_3080P12_3", min_value=0, max_value=100000, value=1)
PECAM1 = st.number_input("PECAM1", min_value=0, max_value=100000, value=5020)
feature_values = [SLC6A13, ANLN, MARCO, SYT13, ARG2, MEFV, ZNF29P, FLVCR2, PTGFR, CRISP2, EME1, IL22RA2,
                  SLC29A4, CYBB, LRRC25, SCN8A, LILRA6, CTD_3080P12_3, PECAM1]
feature = np.array([feature_values])
if st.button("Predict"):
    predicted_class = model.predict(feature)[0]
    predicted_proba = model.predict_proba(feature)[0]
    st.write(f"**Predicted Class:** {predicted_class} (1: Tumor, 0: Normal)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, we're sorry to tell you that you're at high risk of having non - small cell lung cancer. Please contact a professional doctor for a thorough check - up as soon as possible. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    else:
        advice = (f"According to our model, we're glad to inform you that your risk of non - small cell lung cancer is low. But if you feel unwell, consult a professional doctor. Wish you good health. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    st.write(advice)
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_names], colnames=feature_names))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_names], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_names], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP FOrce Plot Explanation')
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Normal', 'Tumor'],
        mode='classification'
        )
    lime_exp = lime_explainer.explain_instance(
        data_row=feature.flatten(),
        predict_fn=model.predict_proba
        )
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)
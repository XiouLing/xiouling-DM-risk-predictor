import streamlit as st
import pandas as pd
import joblib
from llama_cpp import Llama
from transformers import pipeline
st.title("糖尿病風險預測器")
st.markdown("請輸入您的基本資料")
input_data = {
    'Pregnancies': st.number_input("懷孕次數", min_value=0, max_value=17, value=0,key='Pregnancies'),
    'Glucose': st.number_input("葡萄糖濃度", min_value=44, max_value=199, value=120,key='Glucose'),
    'BloodPressure': st.number_input("血壓", min_value=24, max_value=122, value=70,key='BloodPressure'),
    'SkinThickness': st.number_input("皮膚厚度", min_value=7, max_value=99, value=20,key='SkinThickness'),
    'Insulin': st.number_input("胰島素", min_value=14, max_value=846, value=79,key='Insulin'),
    'BMI': st.number_input("BMI", min_value=18.2, max_value=67.1, value=25.6,key='BMI'),
    'DiabetesPedigreeFunction': st.number_input("糖尿病遺傳指數", min_value=0.07, max_value=2.42, value=0.5,key='DiabetesPedigreeFunction'),
    'Age': st.number_input("年齡", min_value=21, max_value=81, value=33,key='Age')
}
if st.button("預測糖尿病風險",key="predict_button"):
    interaction_model=joblib.load("interaction_model.joblib")
    scaler=joblib.load("scaler.joblib")
    top_features=joblib.load("top_features.joblib")
    input_df = pd.DataFrame([input_data])
    for feature in top_features:
        f1,f2=feature.split("*")
        input_df[feature]=input_df[f1]*input_df[f2]
    input_selected=input_df[top_features]
    input_scaler=scaler.transform(input_selected)
    pred=interaction_model.predict(input_scaler)[0]
    prob=interaction_model.predict_proba(input_scaler)[0][1]
    st.subheader("預測結果")
    st.write(f"您有{'**高**'if pred==1 else'**低**'}的糖尿病風險")
    st.write(f"預測機率為:{prob:.2%}")
# ===========================
# AI 健康助理問答功能開始
# ===========================
st.header("AI健康助理問答")
st.markdown("請輸入與健康、糖尿病相關的問題")

user_question = st.text_input("輸入你的問題",key="user_question_input")
if user_question:
    with st.spinner("AI 助理思考中..."):
        llm = Llama(
            model_path = r"C:\Users\User\.lmstudio\models\NousResearch\Nous-Hermes-2-Mistral-7B-DPO-GGUF\Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf",
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0  # 若使用 GPU 可改為 >0
        )
        prompt_template = """
        請根據以下數據回答問題：

        懷孕次數：{pregnancies}
        葡萄糖濃度：{glucose}
        血壓：{blood_pressure}
        皮膚厚度：{skin_thickness}
        胰島素：{insulin}
        BMI：{bmi}
        糖尿病遺傳指數：{genetic_index}
        年齡：{age}

        問題：我今年{age}歲，雖然糖尿病風險不高，但想了解根據我的身體數據是否有需要注意的地方？
        """

        prompt = prompt_template.format(
            pregnancies=input_data['Pregnancies'],
            glucose=input_data['Glucose'],
            blood_pressure=input_data['BloodPressure'],
            skin_thickness=input_data['SkinThickness'],
            insulin=input_data['Insulin'],
            bmi=input_data['BMI'],
            genetic_index=input_data['DiabetesPedigreeFunction'],
            age=input_data['Age']
        )

        output = llm(
            prompt,
            max_tokens=1000,
            stop=["</s>"]
        )
        response_text = ""
        if isinstance(output, dict):
            if 'choices' in output and len(output['choices']) > 0 and 'text' in output['choices'][0]:
                response_text = output['choices'][0]['text']
            elif 'text' in output:
                response_text = output['text']
            else:
                response_text = "AI無法理解您的問題，請再試一次"
        else:
            response_text = str(output)

        st.subheader("AI 助理的回覆")
        st.write(response_text.strip())
    sentiment_pipeline=pipeline("sentiment-analysis")
    result=sentiment_pipeline(user_question)
    st.write("情緒分析結果:",result)
    sentiment_result=sentiment_pipeline(user_question)[0]
    sentiment_label=sentiment_result['label']
    sentiment_score=sentiment_result['score']
    st.write(f"檢測到您目前的情緒傾向為:**{sentiment_label}**，信心水準:{sentiment_score:.2%}")
    if sentiment_label=="NEGATIVE":
        st.info("AI感受到您可能有些負面情緒，希望您一切安好，如果需要協助，也可以尋求專業資源。")
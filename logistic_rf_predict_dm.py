from llama_cpp import Llama
import joblib
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import streamlit as st
from transformers import pipeline
df=pd.read_csv('C:/DATA/Pima Indians Diabetes Dataset.csv')
pd.set_option('display.max_columns',100)#因欄位過多，無法完整呈現，故增加此程式碼，以完整呈現
st.write("描述性統計",df.describe())
x=df.drop('Outcome',axis=1)#x自變項不應包含y依變項
y=df['Outcome']#y依變項只看outcome
correlation_matrix=df.corr()#先探討線性關係
st.write("皮爾森相關係數",correlation_matrix['Outcome'].sort_values(ascending=False))#呈現出X與Y之線性關係，並依據大小排列(由大至小)
#因outcome為二元，皮爾森相關係數雖不大，但結果有顯示出線性關係，故以邏輯斯回歸模型建模
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)#測試集2成、訓練集8成
model=LogisticRegression(max_iter=1000)#設定1000為凝合上限
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
st.write("邏輯斯回歸準確率:",accuracy_score(y_test,y_pred))
st.write("邏輯斯回歸混淆矩陣:",confusion_matrix(y_test,y_pred))
st.write("邏輯斯回歸分類報告:",classification_report(y_test,y_pred))
coeff_df=pd.DataFrame({
    "Feature":x.columns,
    "Coefficient":model.coef_[0]
}).sort_values(by="Coefficient",ascending=False)#將係數由大排到小
st.write("邏輯斯回歸特徵係數解釋:",coeff_df)
y_dm = pd.DataFrame(y.value_counts()).reset_index()
y_dm.columns = ['Outcome', 'Count']  # 設定欄位名稱
st.write("有無糖尿病人數:",y_dm)
#因邏輯斯回歸模型結果顯示，雖有良好準確率，但線性關係仍不夠強，故以隨機森林建模(隨機森林不強調線性關係)
rf_model=RandomForestClassifier(random_state=40)#隨機森林程式碼與邏輯斯回歸類似，皆須分測試、訓練集，因邏輯斯回歸已分過，故隨機森林跳過此段
rf_model.fit(x_train,y_train)
rf_pred=rf_model.predict(x_test)
st.write("隨機森林準確率:",accuracy_score(y_test,rf_pred))
st.write("隨機森林混淆矩陣:\n",confusion_matrix(y_test,rf_pred))
st.write("隨機森林分類報告:\n",classification_report(y_test,rf_pred))
features_importance=pd.DataFrame({
    "Feature":x.columns,
    "Importance":rf_model.feature_importances_
}).sort_values(by="Importance",ascending=False)
st.write("隨機森林特徵重要性:",features_importance)
#找出與outcome線性關係較強的交互作用組合
features=df.drop('Outcome',axis=1).columns
y=df['Outcome']
for f1,f2 in combinations(features,2):
    new_col=f'{f1}*{f2}'
    df[new_col]=df[f1]*df[f2]
interaction_col=[col for col in df.columns if '*' in col]
selected_interations=[]
for col in interaction_col:
    corr,p_value=pearsonr(df[col],y)
    if abs(corr)>=0.3 and p_value<0.05:#係數無論正負>=0.3都值得探討，故取絕對值，且p<0.05也有達到統計上顯著差異，也值得探討
        significant="Yes"if p_value<0.05 else "No"
        selected_interations.append((col,round(corr,3),round(p_value,5),significant))
selected_df=pd.DataFrame(selected_interations,columns=['Feature','Correlation','P-value','Significant']).sort_values(by='Correlation',ascending=False)
st.write("與outcome相關性",selected_df)
top_features=selected_df['Feature'].tolist()#把符合以上條件的(係數及p值)找出來
#將交互作用較強的組合做邏輯斯回歸模型
x=df[top_features]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model_mix=LogisticRegression(max_iter=2000)
model_mix.fit(x_train_scaled,y_train)
y_pred=model_mix.predict(x_test_scaled)
st.write("只用交互作用特徵模型的結果:",classification_report(y_test,y_pred))
coeff_df=pd.DataFrame({
    'Feature':top_features,
    'Coefficient':model_mix.coef_[0]
})
important_interactions=coeff_df[coeff_df['Coefficient'].abs()>1]
important_interactions=important_interactions.sort_values(by='Coefficient',ascending=False)
st.write("在模型中影響較大的交互作用項:",important_interactions)#因為上面都沒列出哪些模型是交互作用較強的組合，所以這裡我想列出來
#使用者互動
# 載入資料
df = pd.read_csv(r"C:\DATA\Pima Indians Diabetes Dataset.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 拆訓練測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# 1. 建立 PolynomialFeatures 物件並 fit_transform 訓練資料
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# 2. 建立 StandardScaler 並 fit_transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)

# 3. 建立邏輯斯回歸模型並 fit
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# 儲存物件
joblib.dump(poly, 'poly.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'logistic.pkl')
st.write("模型與轉換器已儲存")

# 載入物件
poly = joblib.load('poly.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('logistic.pkl')
st.title("糖尿病風險預測器")
st.markdown("請輸入您的基本資料")
input_data = {
    'Pregnancies': st.number_input("懷孕次數", min_value=0, max_value=17, value=0),
    'Glucose': st.number_input("葡萄糖濃度", min_value=44, max_value=199, value=120),
    'BloodPressure': st.number_input("血壓", min_value=24, max_value=122, value=70),
    'SkinThickness': st.number_input("皮膚厚度", min_value=7, max_value=99, value=20),
    'Insulin': st.number_input("胰島素", min_value=14, max_value=846, value=79),
    'BMI': st.number_input("BMI", min_value=18.2, max_value=67.1, value=25.6),
    'DiabetesPedigreeFunction': st.number_input("糖尿病遺傳指數", min_value=0.07, max_value=2.42, value=0.5),
    'Age': st.number_input("年齡", min_value=21, max_value=81, value=33)
}

if st.button("預測糖尿病風險"):
    input_df = pd.DataFrame([input_data])
    poly = joblib.load('poly.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('logistic.pkl')

    input_poly = poly.transform(input_df)
    input_scaled = scaler.transform(input_poly)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("預測結果")
    st.write(f"您有{'**高**' if pred == 1 else '**低**'}的糖尿病風險")
    st.write(f"預測機率為: {prob:.2%}")

# ===========================
# AI 健康助理問答功能開始
# ===========================
st.header("AI健康助理問答")
st.markdown("請輸入與健康、糖尿病相關的問題")

user_question = st.text_input("輸入你的問題")
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
        system_prompt="請以溫和、同理心的語氣回覆使用者。"
    else:
        system_prompt="請簡明、清楚地回覆使用者問題。"
    prompt_template=system_prompt+"\n\n"+prompt_template
def df_to_sql(df,table_name='MyTable'):
    cols=','.join([f"{col}VARCHAR(255)"if df[col].dtype=='object'else f"{col}INT"for col in df.columns])
    create_stmt=f"CREATE TABLE{table_name}({cols})；\n"

    insert_stmts=""
    for _,row in df.iterrows():
        values=",".join([f"'{v}'"if isinstance(v,str)else str(v)for v in row])
        insert_stmts+=f"INSERT INTO {table_name}VALUES ({values})；\n"
    return create_stmt+insert_stmts
print(df_to_sql(df))
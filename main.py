import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from itertools import combinations
from scipy.stats import pearsonr
df=pd.read_csv('C:/DATA/Pima Indians Diabetes Dataset.csv')
pd.set_option('display.max_columns',100)#因欄位過多，無法完整呈現，故增加此程式碼，以完整呈現
print("描述性統計",df.describe())
x=df.drop('Outcome',axis=1)#x自變項不應包含y依變項
y=df['Outcome']#y依變項只看outcome
correlation_matrix=df.corr()#先探討線性關係
print("皮爾森相關係數",correlation_matrix['Outcome'].sort_values(ascending=False))#呈現出X與Y之線性關係，並依據大小排列(由大至小)
#因outcome為二元，皮爾森相關係數雖不大，但結果有顯示出線性關係，故以邏輯斯回歸模型建模
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
def train_logistic_regression(x_train,x_test,y_train,y_test):
    model=LogisticRegression(max_iter=1000)#設定1000為凝合上限
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    report=classification_report(y_test,y_pred,output_dict=True)
    print("Logistic Regression Accuracy",accuracy_score(y_test,y_pred))
    print("Logistic Regression Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    print("Logistic Regression Classification Report:\n",report)
    coef_df=pd.DataFrame({
        'Feature':x_train.columns,
        'Coefficient':model.coef_[0]
    }).sort_values(by='Coefficient',ascending=False)
    print("Coefficients:\n",coef_df)
    print("\n")
#因邏輯斯回歸模型結果顯示，雖有良好準確率，但線性關係仍不夠強，故以隨機森林建模(隨機森林不強調線性關係)
def train_random_forest(x_train,x_test,y_train,y_test):
    model=RandomForestClassifier(random_state=40)#隨機森林程式碼與邏輯斯回歸類似，皆須分測試、訓練集，因邏輯斯回歸已分過，故隨機森林跳過此段
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    report=classification_report(y_test,y_pred)
    print("Random Forest Accuracy:",accuracy_score(y_test,y_pred))
    print("Random Forest Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    print("Random Forest Classification Report:\n",report)
    imp_df=pd.DataFrame({
        'Feature':x_train.columns,
        'Importance':model.feature_importances_
    }).sort_values(by='Importance',ascending=False)
    print("Feature Importance:\n",imp_df)
    print("\n")
#找出與outcome線性關係較強的交互作用組合
def generate_interaction_terms(df,y_col='Outcome'):
    features=df.drop(y_col,axis=1).columns
    y=df[y_col]
    for f1,f2 in combinations(features,2):
        df[f'{f1}*{f2}']=df[f1]*df[f2]
    interaction_cols=[col for col in df.columns if '*'in col]
    selected=[]
    for col in interaction_cols:
        corr,p=pearsonr(df[col],y)
        if abs(corr)>=0.3 and p<0.05:
            selected.append(col)
    return selected
#將交互作用較強的組合做邏輯斯回歸模型
def train_interaction_logistic_model(df,top_features,test_size=0.3,random_state=40):
    for feature in top_features:
        f1,f2=feature.split("*")
        df[feature]=df[f1]*df[f2]
    x=df[top_features]
    y=df['Outcome']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    model=LogisticRegression(max_iter=2000)
    model.fit(x_train_scaled,y_train)
    y_pred=model.predict(x_test_scaled)
    report=classification_report(y_test,y_pred)
    print("Interaction Logistic Regression Accuracy:",accuracy_score(y_test,y_pred))
    print("Interaction Logistic Regression Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    print("Interaction Logistic Regression Classification Report:\n",report)
    coeff_df=pd.DataFrame({
        'Feature':top_features,
        'Coefficient':model.coef_[0]
    }).sort_values(by='Coefficient',ascending=False)
    print("Coefficient:\n",coeff_df)
    print("\n")
    return model,scaler,top_features
train_logistic_regression(x_train,x_test,y_train,y_test)
train_random_forest(x_train,x_test,y_train,y_test)
top_features=generate_interaction_terms(df)
interaction_model,scaler,top_features=train_interaction_logistic_model(df.copy(),top_features)
import joblib
joblib.dump(interaction_model,"interaction_model.joblib")
joblib.dump(scaler,"scaler.joblib")
joblib.dump(top_features,"top_features.joblib")
print("第三個模型已儲存為interaction_model.joblib等檔案")
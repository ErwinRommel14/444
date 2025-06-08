import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(page_title="Disney Princess Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #333333;'>🎀 Disney Princess Iconic Status Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Прогнозируем, станет ли принцесса культовой на основе характеристик фильма.</p>", unsafe_allow_html=True)

# Загрузка данных
@st.cache_data
def load_data():
    url = "disney_princess_popularity_dataset_300_rows.csv"
    return pd.read_csv(url)

try:
    data = load_data()
except:
    st.error("Ошибка загрузки данных. Используем демо-данные.")
    data = pd.DataFrame({
        'NumberOfSongs': [4, 1, 3, 5],
        'HasSoloSong': ['Yes', 'No', 'Yes', 'No'],
        'BoxOfficeMillions': [500, 300, 700, 200],
        'IMDB_Rating': [7.5, 6.8, 8.2, 6.5],
        'IsIconic': ['Yes', 'No', 'Yes', 'No']
    })

# Предобработка данных
def preprocess_data(df):
    df = df.copy()

    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['HasSoloSong', 'HasDuet', 'IsRoyalByBirth', 'HasAnimalSidekick',
                  'HasMagicalPowers', 'SpeaksToAnimals', 'FightsVillainDirectly', 'IsIconic']

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    num_cols = ['NumberOfSongs', 'BoxOfficeMillions', 'IMDB_Rating', 'MovieRuntimeMinutes']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

data_processed = preprocess_data(data)

# Выбор фичей
default_features = ['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating']
available_features = [col for col in default_features if col in data_processed.columns]

if not available_features:
    st.error("Нет подходящих признаков для обучения модели.")
    st.stop()

X = data_processed[available_features]
y = data_processed.get('IsIconic', pd.Series(np.zeros(len(data_processed))))

# Разделение данных
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except:
    st.error("Ошибка при разделении данных")
    st.stop()

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
@st.cache_resource
def train_models():
    models = {}

    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        models['Logistic Regression'] = lr
    except Exception as e:
        st.warning(f"Logistic Regression error: {str(e)}")

    try:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
    except Exception as e:
        st.warning(f"Random Forest error: {str(e)}")

    return models

models = train_models()

if not models:
    st.error("Не удалось обучить ни одну модель")
    st.stop()

# Интерфейс пользователя
st.markdown("### 🎫 Параметры принцессы")
def get_user_input():
    inputs = {}

    if 'NumberOfSongs' in available_features:
        inputs['NumberOfSongs'] = st.slider("🎵 Количество песен", 0, 10, 3)

    if 'HasSoloSong' in available_features:
        inputs['HasSoloSong'] = st.radio("🎤 Сольная песня?", ["No", "Yes"]) == "Yes"

    if 'BoxOfficeMillions' in available_features:
        inputs['BoxOfficeMillions'] = st.slider("💰 Кассовые сборы ($ млн)", 0, 2000, 500)

    if 'IMDB_Rating' in available_features:
        inputs['IMDB_Rating'] = st.slider("⭐ Рейтинг IMDB", 0.0, 10.0, 7.0)

    return pd.DataFrame([inputs])

user_input = get_user_input()

# Отображение ввода
st.markdown("### 📋 Введенные параметры")
st.dataframe(user_input.style.highlight_max(axis=0))

# Предсказание
st.markdown("### 🔍 Результат предсказания")

model_choice = st.selectbox("🧠 Выберите модель", list(models.keys()))

if st.button("🔮 Прогнозировать статус"):
    model = models[model_choice]

    try:
        if model_choice == 'Logistic Regression':
            input_scaled = scaler.transform(user_input)
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)
        else:
            prediction = model.predict(user_input)
            proba = model.predict_proba(user_input)

        result = "🌟 Иконная" if prediction[0] == 1 else "💫 Не иконная"
        probability = f"{max(proba[0]) * 100:.1f}%"

        # Нейтральный стиль для прогноза
        st.markdown(f"""
        <div style='background-color:#eaeaea;padding:12px;border-radius:10px;text-align:center;'>
            <h3 style='color:#333;margin:0;'>{result}</h3>
            <p style='margin:5px 0;font-size:14px;'>Вероятность: <strong>{probability}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Маленький график
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.barplot(x=['Не иконная', 'Иконная'], y=proba[0], palette="Blues_r", ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Ошибка предсказания: {str(e)}")

# Оценка моделей
st.markdown("### 📊 Точность моделей")
metrics = []
for name, model in models.items():
    if name == 'Logistic Regression':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    metrics.append({"Модель": name, "Точность": f"{acc:.2%}"})

st.table(pd.DataFrame(metrics).style.set_properties(**{'text-align': 'center'}))

# Важность признаков
if 'Random Forest' in models:
    st.markdown("### 🧩 Важность признаков (Random Forest)")

    try:
        importances = models['Random Forest'].feature_importances_
        feat_importances = pd.Series(importances, index=available_features)
        fig, ax = plt.subplots(figsize=(4, 2))
        feat_importances.nlargest(10).plot(kind='barh', ax=ax, color="#4682B4")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_title("Важность признаков", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Не удалось показать важность признаков: {str(e)}")

# Советы по улучшению
st.markdown("### 💡 Как сделать принцессу иконической?")
st.markdown("""
- 🎶 Добавьте больше песен
- 🎤 Дайте ей сольную композицию
- 💰 Стремитесь к высоким сборам (> $500 млн)
- ⭐ Поддерживайте рейтинг выше 7.5
""")

# Финальный стиль
st.markdown("""
<style>
    .stButton button {
        background-color: #4682B4;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)
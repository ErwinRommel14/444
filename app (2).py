import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы
st.set_page_config(
    page_title="Disney Princess Classifier", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4682B4;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a6d99;
        transform: scale(1.02);
    }
    .prediction-box {
        background-color: #f0f0f0;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("<h1 style='text-align: center; color: #FFD700;'>✨ Disney Princess Iconic Status Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Прогнозируем, станет ли принцесса культовой на основе характеристик фильма.</p>", unsafe_allow_html=True)

# Загрузка данных
@st.cache_data
def load_data():
    try:
        data = pd.DataFrame({
            'NumberOfSongs': [4, 1, 3, 5, 6, 2, 4, 3],
            'HasSoloSong': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
            'BoxOfficeMillions': [500, 300, 700, 200, 800, 350, 600, 400],
            'IMDB_Rating': [7.5, 6.8, 8.2, 6.5, 8.5, 7.0, 7.8, 6.9],
            'IsIconic': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
        })
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.error("Данные не загружены. Пожалуйста, проверьте источник данных.")
    st.stop()

# Предобработка данных
def preprocess_data(df):
    df = df.copy()
    binary_map = {'Yes': 1, 'No': 0}
    df['HasSoloSong'] = df['HasSoloSong'].map(binary_map).fillna(0)
    df['IsIconic'] = df['IsIconic'].map(binary_map).fillna(0)
    return df

data_processed = preprocess_data(data)

# Обучение модели
@st.cache_resource
def train_model():
    try:
        X = data_processed[['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating']]
        y = data_processed['IsIconic']
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Ошибка обучения модели: {str(e)}")
        return None

model = train_model()

if model is None:
    st.stop()

# Ввод параметров
st.markdown("### 🎫 Параметры принцессы")
inputs = {}
inputs['NumberOfSongs'] = st.slider("🎵 Количество песен", 0, 10, 3)
inputs['HasSoloSong'] = st.radio("🎤 Сольная песня?", ["No", "Yes"]) == "Yes"
inputs['BoxOfficeMillions'] = st.slider("💰 Кассовые сборы ($ млн)", 0, 2000, 500)
inputs['IMDB_Rating'] = st.slider("⭐ Рейтинг IMDB", 0.0, 10.0, 7.0, step=0.1)

user_input = pd.DataFrame([inputs])

# Предсказание
if st.button("🔮 Прогнозировать статус", key="predict_button"):
    try:
        prediction = model.predict(user_input)
        proba = model.predict_proba(user_input)

        result = "🌟 Иконная" if prediction[0] == 1 else "💫 Не иконная"
        probability = f"{max(proba[0]) * 100:.1f}%"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style='color:#333;margin:0;'>{result}</h3>
            <p style='margin:5px 0;font-size:14px;'>Вероятность: <strong>{probability}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # График вероятностей
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=['Не иконная', 'Иконная'], y=proba[0], palette="Blues_r", ax=ax)
        ax.set_ylabel("Вероятность")
        ax.set_title("Распределение вероятностей", fontsize=12)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
        
    except Exception as e:
        st.error(f"⚠️ Ошибка предсказания: {str(e)}")

# Советы по улучшению
st.markdown("### 💡 Как сделать принцессу иконической?")
st.markdown("""
- 🎶 **Добавьте больше песен** (идеально 4+)
- 🎤 **Сольная песня** значительно увеличивает шансы
- 💰 **Кассовые сборы** > $500 млн
- ⭐ **Рейтинг IMDB** выше 7.5
- 🦸 **Сильный характер** и самостоятельность
""")
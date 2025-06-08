import streamlit as st

# Конфигурация страницы (должна быть первой)
st.set_page_config(
    page_title="Disney Princess Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Стили CSS
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        overflow: hidden;
    }
    .stPlotContainer {
        position: relative;
    }
    .result-box {
        background:#f8f9fa;
        padding:1.5rem;
        border-radius:10px;
        text-align:center;
        margin:1rem 0;
    }
    .result-text {
        color: black !important;
        font-weight: bold;
        margin: 0;
    }
    .probability-text {
        color: black !important;
        margin:5px 0;
        font-size:14px;
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Загрузка данных
@st.cache_data
def load_data():
    return pd.DataFrame({
        'NumberOfSongs': [4, 1, 3, 5, 6, 2, 4, 3],
        'HasSoloSong': [1, 0, 1, 0, 1, 0, 1, 0],
        'BoxOfficeMillions': [500, 300, 700, 200, 800, 350, 600, 400],
        'IMDB_Rating': [7.5, 6.8, 8.2, 6.5, 8.5, 7.0, 7.8, 6.9],
        'IsIconic': [1, 0, 1, 0, 1, 0, 1, 0]
    })

data = load_data()

# Обучение модели
@st.cache_resource
def train_model():
    X = data[['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating']]
    y = data['IsIconic']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Интерфейс
st.title("✨ Disney Princess Predictor")

# Ввод параметров
col1, col2 = st.columns(2)
with col1:
    songs = st.slider("🎵 Количество песен", 0, 10, 3)
    solo = st.radio("🎤 Сольная песня?", ["Нет", "Да"]) == "Да"
with col2:
    box_office = st.slider("💰 Кассовые сборы ($ млн)", 0, 2000, 500)
    rating = st.slider("⭐ Рейтинг IMDB", 0.0, 10.0, 7.0, step=0.1)

# Предсказание
if st.button("🔮 Прогнозировать статус"):
    input_data = pd.DataFrame([[songs, solo, box_office, rating]], 
                            columns=['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating'])
    
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        # Результат с черным цветом текста
        result = "🌟 Иконная" if prediction == 1 else "💫 Не иконная"
        st.markdown(f"""
        <div class="result-box">
            <h3 class="result-text">{result}</h3>
            <p class="probability-text">Вероятность: <strong>{max(proba)*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # График
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=['Не иконная', 'Иконная'], y=proba, palette="viridis", ax=ax)
        ax.set_ylabel("Вероятность")
        plt.tight_layout()
        
        # Используем буфер для избежания ошибок DOM
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
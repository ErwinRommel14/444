import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы (должна быть первой)
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
    .result-text {
        color: black !important;
        font-weight: bold;
    }
    .probability-text {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.title("✨ Disney Princess Iconic Status Predictor")

# Загрузка данных
@st.cache_data
def load_data():
    try:
        # Используем предоставленный файл данных
        data = pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        # Резервные данные на случай ошибки
        return pd.DataFrame({
            'NumberOfSongs': [4, 1, 3, 5],
            'HasSoloSong': ['Yes', 'No', 'Yes', 'No'],
            'BoxOfficeMillions': [500, 300, 700, 200],
            'IMDB_Rating': [7.5, 6.8, 8.2, 6.5],
            'IsIconic': ['Yes', 'No', 'Yes', 'No']
        })

data = load_data()

# Проверка данных
if data.empty:
    st.error("Данные не загружены. Пожалуйста, проверьте файл данных.")
    st.stop()

# Предобработка данных
def preprocess_data(df):
    df = df.copy()
    
    # Преобразование бинарных признаков
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['HasSoloSong', 'IsIconic']  # Добавьте другие бинарные колонки при необходимости
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)
    
    # Обработка числовых признаков
    num_cols = ['NumberOfSongs', 'BoxOfficeMillions', 'IMDB_Rating']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    
    return df

data_processed = preprocess_data(data)

# Обучение модели
@st.cache_resource
def train_model():
    try:
        # Используем признаки из вашего датасета
        features = ['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating']
        X = data_processed[features]
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

# Интерфейс пользователя
st.markdown("### 🎫 Параметры принцессы")

# Ввод параметров
col1, col2 = st.columns(2)
with col1:
    songs = st.slider("🎵 Количество песен", 0, 10, 3)
    solo = st.radio("🎤 Сольная песня?", ["No", "Yes"]) == "Yes"
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
        
        # Результат
        result = "🌟 Иконная" if prediction == 1 else "💫 Не иконная"
        probability = f"{max(proba)*100:.1f}%"
        
        st.markdown(f"""
        <div style='background:#f8f9fa;padding:1.5rem;border-radius:10px;text-align:center;margin:1rem 0;'>
            <h3 class="result-text">{result}</h3>
            <p class="probability-text">Вероятность: <strong>{probability}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # График вероятностей
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=['Не иконная', 'Иконная'], y=proba, palette="viridis", ax=ax)
        ax.set_ylabel("Вероятность")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ Ошибка предсказания: {str(e)}")

# Информация о датасете
st.markdown("---")
st.markdown("### 📊 Информация о датасете")
st.write(f"Всего записей: {len(data)}")
st.write(f"Пример данных:")
st.dataframe(data.head(3))

# Советы по улучшению
st.markdown("### 💡 Как сделать принцессу иконической?")
st.markdown("""
- 🎶 **Добавьте больше песен** (идеально 4+)
- 🎤 **Сольная песня** значительно увеличивает шансы
- 💰 **Кассовые сборы** > $500 млн
- ⭐ **Рейтинг IMDB** выше 7.5
""")
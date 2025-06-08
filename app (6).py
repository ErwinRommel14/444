import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы
st.set_page_config(page_title="Disney Princess Predictor", layout="centered")

st.title("✨ Disney Princess Iconic Status Predictor")

# --- КЭШИРОВАННЫЕ ФУНКЦИИ ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")
        return df.copy()
    except FileNotFoundError:
        st.warning("Файл не найден. Используются тестовые данные.")
        return pd.DataFrame({
            'NumberOfSongs': [4, 1, 3, 5],
            'HasSoloSong': ['Yes', 'No', 'Yes', 'No'],
            'BoxOfficeMillions': [500, 300, 700, 200],
            'IMDB_Rating': [7.5, 6.8, 8.2, 6.5],
            'IsIconic': ['Yes', 'No', 'Yes', 'No']
        })

@st.cache_resource
def preprocess_and_train():
    # Загрузка данных
    df = load_data()

    # Предобработка
    df = df.copy()
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['HasSoloSong', 'IsIconic']:
        df[col] = df[col].map(binary_map).fillna(0)
    for col in ['NumberOfSongs', 'BoxOfficeMillions', 'IMDB_Rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Обучение модели
    features = ['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating']
    X = df[features]
    y = df['IsIconic']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return df, model

# --- ЗАГРУЖАЕМ ВСЁ ОДНИМ РАЗОМ ---

try:
    data_processed, model = preprocess_and_train()
except Exception as e:
    st.error(f"Ошибка загрузки данных или модели: {e}")
    st.stop()

# --- ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ---

st.markdown("### 🎫 Параметры принцессы")

col1, col2 = st.columns(2)
with col1:
    songs = st.slider("🎵 Количество песен", 0, 10, 3)
    solo = st.radio("🎤 Сольная песня?", ["No", "Yes"]) == "Yes"
with col2:
    box_office = st.slider("💰 Кассовые сборы ($ млн)", 0, 2000, 500)
    rating = st.slider("⭐ Рейтинг IMDB", 0.0, 10.0, 7.0, step=0.1)

if st.button("🔮 Прогнозировать статус"):
    input_data = pd.DataFrame([[songs, solo, box_office, rating]],
                              columns=['NumberOfSongs', 'HasSoloSong', 'BoxOfficeMillions', 'IMDB_Rating'])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    result_text = "🌟 Иконная принцесса" if prediction == 1 else "💫 Не иконная принцесса"
    probability = f"{max(proba) * 100:.1f}%"

    # Вывод результата через Streamlit компоненты
    st.markdown("### 📌 Результат прогноза")
    if prediction == 1:
        st.success(f"{result_text} ✨")
    else:
        st.info(f"{result_text} 💫")

    st.progress(int(max(proba) * 100))
    st.caption(f"Вероятность: **{probability}**")

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

# --- Информация о датасете ---
if st.checkbox("🔍 Показать пример данных"):
    st.markdown("### 📊 Информация о датасете")
    st.write(f"Всего записей: {len(data_processed)}")
    st.dataframe(data_processed.head(3))

# --- Советы ---
st.markdown("### 💡 Как сделать принцессу иконической?")
st.markdown("""
- 🎶 **Добавьте больше песен** (идеально 4+)
- 🎤 **Сольная песня** значительно увеличивает шансы
- 💰 **Кассовые сборы** > $500 млн
- ⭐ **Рейтинг IMDB** выше 7.5
""")
# -*- coding: utf-8 -*-
import sys
import subprocess
import importlib.util

# Проверяем наличие необходимных библиотек и устанавливаем их при необходимости
def install_packages():
    required_packages = ['streamlit', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            st.info(f'Установка пакета: {package}...')
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success(f"Пакет {package} успешно установлен!")

# Вызываем установку пакетов
install_packages()

# Теперь импортируем после установки
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Кэширование модели
@st.cache_data
def train_model():
    try:
        df = pd.read_csv('bots_vs_users.csv')
    except FileNotFoundError:
        st.error("Файл 'bots_vs_users.csv' не найден. Проверьте его наличие в рабочей директории.")
        return None, None, 0, {}

    # Предобработка данных
    df_v1 = df.loc[:, df.isna().mean() <= 0.5]
    subscribers_target = df_v1[['target', 'subscribers_count']]
    other_features = df_v1.drop(['target', 'subscribers_count'], axis=1)
    other_features = other_features.replace('Unknown', 3)
    df_v2 = pd.concat([subscribers_target, other_features], axis=1)
    df_v2 = df_v2.replace('Unknown', np.nan)
    df_v2['city'] = df_v2['city'].apply(lambda x: 0 if x == 3 else 1)
    df_v3 = df_v2.apply(lambda col: col.fillna(col.mode()[0]))
    df_final = df_v3.astype(float).astype(int)

    # Удаление выбросов
    Q1 = df_final['subscribers_count'].quantile(0.25)
    Q3 = df_final['subscribers_count'].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    max_limit = Q3 + 1.5 * IQR
    data = df_final[(df_final['subscribers_count'] >= low_limit) &
                    (df_final['subscribers_count'] <= max_limit)]

    X = data.drop(['target'], axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', KNeighborsClassifier())
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Перестановочная важность
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
    feature_weights = result.importances_mean

    return pipeline, X.columns, accuracy, report, feature_weights

# Функция для предсказания
def predict_bot(pipeline, features, input_values):
    input_df = pd.DataFrame([input_values], columns=features)
    scaled_input = pipeline.named_steps['scaler'].transform(input_df)
    return pipeline.named_steps['classifier'].predict_proba(scaled_input)[0][1]

# Основная функция
def main():
    st.markdown('<p class="header-style">BotDetector PRO</p>', unsafe_allow_html=True)
    st.caption("Система обнаружения ботов в социальных сетях")

    with st.spinner('Загрузка модели детектора ботов...'):
        result = train_model()
        if result is None:
            return
        pipeline, features, accuracy, report, feature_weights = result

    # Форма ввода параметров
    with st.form("bot_form"):
        st.subheader("Параметры аккаунта")
        input_values = {}
        cols = st.columns(2)

        with cols[0]:
            input_values['subscribers_count'] = st.number_input("Количество подписчиков", min_value=0, value=500, key='subs')
            input_values['has_avatar'] = st.selectbox("Аватарка", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет", key='avatar')
            input_values['has_cover'] = st.selectbox("Обложка профиля", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет", key='cover')
            input_values['city'] = st.selectbox("Город указан", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет", key='city')

        with cols[1]:
            input_values['verified'] = st.selectbox("Верифицирован", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет", key='verified')
            input_values['posts_count'] = st.number_input("Количество постов", min_value=0, value=100, key='posts')
            input_values['followers_to_following'] = st.number_input("Соотношение подписчики/подписки", min_value=0.0, value=1.5, key='ratio')
            input_values['digits_in_username'] = st.number_input("Цифры в имени", min_value=0, value=2, key='digits')

        submitted = st.form_submit_button("Проверить аккаунт", type="primary")

    result_display = st.empty()

    if len(input_values) == len(features):
        try:
            bot_prob = predict_bot(pipeline, features, input_values)
            threshold = 0.5
            status = "БОТ" if bot_prob >= threshold else "Настоящий"
            status_color = "#e74c3c" if bot_prob >= threshold else "#2ecc71"
            pulse_class = "pulse-animation" if bot_prob >= threshold else ""

            with result_display.container():
                st.subheader("Результат проверки")
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"""
                    <div class="bot-indicator {pulse_class}" style="margin-bottom: 1rem;">
                        <div style="font-size: 16px; color: #7f8c8d;">Вероятность бота</div>
                        <div class="metric-value" style="color: {status_color};">{bot_prob*100:.1f}%</div>
                        <div style="font-size: 14px; color: #7f8c8d;">Порог: {threshold*100:.0f}%</div>
                        <div style="font-size: 14px; color: {status_color}; font-weight: bold;">Статус: {status}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    progress_bar = st.progress(0)
                    bot_prob_clipped = min(max(bot_prob, 0), 1)
                    for percent_complete in range(int(bot_prob_clipped * 100)):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    progress_bar.progress(bot_prob_clipped)

                feat_importance = pd.DataFrame({
                    'Признак': features,
                    'Важность': feature_weights
                }).sort_values('Важность', ascending=False)

                plt.figure(figsize=(10, 6))
                sns.barplot(x='Важность', y='Признак', data=feat_importance, palette='viridis')
                plt.title('Важность признаков для определения ботов')
                plt.tight_layout()
                st.pyplot(plt)

        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")

    if submitted:
        st.success("✅ Анализ завершен! Результаты обновлены выше.")
        if bot_prob >= threshold:
            st.warning("⚠️ Внимание! Обнаружен высокий риск бота.")
        else:
            st.balloons()

    # Боковая панель
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.info("BotDetector PRO использует KNN-алгоритм для анализа аккаунтов в соцсетях на признаки ботов.")

if __name__ == "__main__":
    main()

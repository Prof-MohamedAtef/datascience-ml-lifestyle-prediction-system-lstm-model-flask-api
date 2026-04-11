import streamlit as st
import pandas as pd

df = pd.read_csv('D:/PhDResearch/qualificationExams/2026/selectedTopics2/anaconda/dataset/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv')

st.title("AI Lifestyle Dashboard")

st.line_chart(df['TotalSteps'])

st.bar_chart(df[['Calories', 'VeryActiveMinutes']])
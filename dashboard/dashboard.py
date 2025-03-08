import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

DATA_PATH = "dashboard/main_data.csv"
df_all = pd.read_csv(DATA_PATH)

df_all["datetime"] = pd.to_datetime(df_all[["year", "month", "day", "hour"]])
df_all["quarter"] = df_all["datetime"].dt.to_period("Q")

st.sidebar.title("Dashboard Kualitas Udara")
pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
selected_pollutant = st.sidebar.selectbox("Pilih Polutan", pollutants)
st.title("Distribusi Persebaran Polutan dari Berbagai Stasiun di China")

st.subheader("Statistik Data Kualitas Udara")

col1, col2, col3 = st.columns(3)

total_stations = df_all["Station"].nunique()
total_per_station = df_all.groupby("Station").size().mean()
total_data = len(df_all)

with col1:
    st.metric("Total Stasiun", value=total_stations)

with col2:
    st.metric("Rata-rata Data per Stasiun", value=int(total_per_station))

with col3:
    st.metric("Total Data Keseluruhan", value=total_data)

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(df_all["Station"].value_counts().index, df_all["Station"].value_counts().values, color="#90CAF9")
ax.set_xlabel("Stasiun", fontsize=14)
ax.set_ylabel("Jumlah Data", fontsize=14)
ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)

#Permasalahan 1: Pola distribusi polutan di berbagai lokasi berdasarkan analisis geospasial
st.title("Distribusi Polutan di Berbagai Lokasi")

station_coordinates = {
    "Aotizhongxin": (39.95, 116.41),
    "Changping": (40.22, 116.23),
    "Dingling": (40.29, 116.22),
    "Dongsi": (39.93, 116.42),
    "Guanyuan": (39.93, 116.36),
    "Gucheng": (39.91, 116.30),
    "Huairou": (40.32, 116.63),
    "Nongzhanguan": (39.93, 116.47),
    "Shunyi": (40.12, 116.66),
    "Tiantian": (39.99, 116.38),
    "Wanliu": (39.98, 116.30),
    "Wanshouxigong": (39.88, 116.35),
}

df_all["Latitude"] = df_all["Station"].map(lambda x: station_coordinates.get(x, (None, None))[0])
df_all["Longitude"] = df_all["Station"].map(lambda x: station_coordinates.get(x, (None, None))[1])

geo_data = df_all.groupby("Station", as_index=False).agg({
    "Latitude": "first",
    "Longitude": "first",
    selected_pollutant: "mean"
})

fig = px.scatter_mapbox(
    geo_data, lat="Latitude", lon="Longitude",
    hover_name="Station", color=selected_pollutant,
    size=selected_pollutant, zoom=9, mapbox_style="carto-positron",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig)


#Permasalahan 2: Pola konsentrasi polutan berdasarkan waktu menggunakan analisis clustering
st.title("Heatmap Konsentrasi PM2.5 per Kuartal")

df_quarterly = df_all.groupby(["Station", "quarter"])["PM2.5"].mean().unstack(level=0)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df_quarterly.T, cmap="coolwarm", robust=True, ax=ax)
st.pyplot(fig)

#Perbandingan antar 2 stasiun dengan tingkat polutan terendah dan tertinggi
st.title("Perbandingan Polutan Antar Stasiun")
stasiun1 = st.selectbox("Pilih Stasiun 1", df_all["Station"].unique(), index=7)
stasiun2 = st.selectbox("Pilih Stasiun 2", df_all["Station"].unique(), index=2)

df_stasiun1 = df_all[df_all["Station"] == stasiun1].groupby("quarter")["PM2.5"].mean()
df_stasiun2 = df_all[df_all["Station"] == stasiun2].groupby("quarter")["PM2.5"].mean()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_stasiun1.index.astype(str),
    y=df_stasiun1.values,
    mode='lines+markers',
    name=stasiun1,
    line=dict(color='red')  
))

fig.add_trace(go.Scatter(
    x=df_stasiun2.index.astype(str),
    y=df_stasiun2.values,
    mode='lines+markers',
    name=stasiun2,
    line=dict(color='blue')  
))

fig.update_layout(
    title=f"Perbandingan PM2.5 antara {stasiun1} dan {stasiun2}",
    xaxis_title="Kuartal",
    yaxis_title="PM2.5"
)

st.plotly_chart(fig)

# Permasalahan 3: Penerapan teknik RFM (Recency, Frequency, Monetary) dalam analisis kualitas udara
def create_rfm_airquality(df):
    latest_date = df["datetime"].max()
    recency = df.groupby("Station")["datetime"].max().apply(lambda x: (latest_date - x).days)
    frequency = df.groupby("Station")["datetime"].count()
    monetary = df.groupby("Station")["PM2.5"].mean()
    return pd.DataFrame({"Recency": recency, "Frequency": frequency, "Monetary [PM2.5]": monetary})

rfm_df = create_rfm_airquality(df_all)
st.title("Analisis RFM Kualitas Udara")
st.dataframe(rfm_df.style.background_gradient(cmap="coolwarm"))
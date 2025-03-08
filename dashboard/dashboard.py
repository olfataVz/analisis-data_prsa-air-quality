import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

aotizhongxin_df= pd.read_csv("data\\PRSA_Data_Aotizhongxin_20130301-20170228.csv")
aotizhongxin_df.head()

changping_df = pd.read_csv("data\\PRSA_Data_Changping_20130301-20170228.csv")
changping_df.head()

dingling_df = pd.read_csv("data\\PRSA_Data_Dingling_20130301-20170228.csv")
dingling_df.head()

dongsi_df = pd.read_csv("data\\PRSA_Data_Dongsi_20130301-20170228.csv")
dongsi_df.head()

guanyuan_df = pd.read_csv("data\\PRSA_Data_Guanyuan_20130301-20170228.csv")
guanyuan_df.head()

gucheng_df = pd.read_csv("data\\PRSA_Data_Gucheng_20130301-20170228.csv")
gucheng_df.head()

huairou_df = pd.read_csv("data\\PRSA_Data_Huairou_20130301-20170228.csv")
huairou_df.head()

nongzhanguan_df = pd.read_csv("data\\PRSA_Data_Nongzhanguan_20130301-20170228.csv")
nongzhanguan_df.head()

shunyi_df = pd.read_csv("data\\PRSA_Data_Shunyi_20130301-20170228.csv")
shunyi_df.head()

tiantian_df = pd.read_csv("data\\PRSA_Data_Tiantan_20130301-20170228.csv")
tiantian_df.head()

wanliu_df = pd.read_csv("data\\PRSA_Data_Wanliu_20130301-20170228.csv")
wanliu_df.head()

wanshouxigong_df = pd.read_csv("data\\PRSA_Data_Wanshouxigong_20130301-20170228.csv")
wanshouxigong_df.head()
#------------------------------------------------
#------------------------------------------------

aotizhongxin_df.info()
print(f"\nMissing Value:\n{aotizhongxin_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {aotizhongxin_df.duplicated().sum()}")

changping_df.info()
print(f"\nMissing Value:\n{changping_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {changping_df.duplicated().sum()}")

dingling_df.info()
print(f"\nMissing Value:\n{dingling_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {dingling_df.duplicated().sum()}")

dongsi_df.info()
print(f"\nMissing Value:\n{dongsi_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {dongsi_df.duplicated().sum()}")

guanyuan_df.info()
print(f"\nMissing Value:\n{guanyuan_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {guanyuan_df.duplicated().sum()}")

gucheng_df.info()
print(f"\nMissing Value:\n{gucheng_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {gucheng_df.duplicated().sum()}")

huairou_df.info()
print(f"\nMissing Value:\n{huairou_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {huairou_df.duplicated().sum()}")

nongzhanguan_df.info()
print(f"\nMissing Value:\n{nongzhanguan_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {nongzhanguan_df.duplicated().sum()}")

shunyi_df.info()
print(f"\nMissing Value:\n{shunyi_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {shunyi_df.duplicated().sum()}")

tiantian_df.info()
print(f"\nMissing Value:\n{tiantian_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {tiantian_df.duplicated().sum()}")

wanliu_df.info()
print(f"\nMissing Value:\n{wanliu_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {wanliu_df.duplicated().sum()}")

wanshouxigong_df.info()
print(f"\nMissing Value:\n{wanshouxigong_df.isna().sum()}")
print(f"\nTotal Duplikasi Data: {wanshouxigong_df.duplicated().sum()}")

#------------------------------------------------
#------------------------------------------------

aotizhongxin_df['TEMP'] = aotizhongxin_df['TEMP'].fillna(aotizhongxin_df['TEMP'].median())
aotizhongxin_df['PRES'] = aotizhongxin_df['PRES'].fillna(aotizhongxin_df['PRES'].median())
aotizhongxin_df['DEWP'] = aotizhongxin_df['DEWP'].fillna(aotizhongxin_df['DEWP'].median())
aotizhongxin_df['RAIN'] = aotizhongxin_df['RAIN'].fillna(0) 
aotizhongxin_df['WSPM'] = aotizhongxin_df['WSPM'].fillna(aotizhongxin_df['WSPM'].median())

aotizhongxin_df['datetime'] = pd.to_datetime(aotizhongxin_df[['year', 'month', 'day', 'hour']])
aotizhongxin_df.set_index('datetime', inplace=True)
cols_to_interpolate = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
aotizhongxin_df[cols_to_interpolate] = aotizhongxin_df[cols_to_interpolate].interpolate(method='time')
aotizhongxin_df['wd'] = aotizhongxin_df['wd'].fillna(aotizhongxin_df['wd'].mode()[0])

print(f"\nMissing Value:\n{aotizhongxin_df.isna().sum()}")
aotizhongxin_df.describe()
aotizhongxin_df.sample(5)


changping_df['TEMP'] = changping_df['TEMP'].fillna(changping_df['TEMP'].median())
changping_df['PRES'] = changping_df['PRES'].fillna(changping_df['PRES'].median())
changping_df['DEWP'] = changping_df['DEWP'].fillna(changping_df['DEWP'].median())
changping_df['RAIN'] = changping_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
changping_df['WSPM'] = changping_df['WSPM'].fillna(changping_df['WSPM'].median())

changping_df['datetime'] = pd.to_datetime(changping_df[['year', 'month', 'day', 'hour']])
changping_df.set_index('datetime', inplace=True)
changping_df[cols_to_interpolate] = changping_df[cols_to_interpolate].interpolate(method='time')
changping_df['wd'] = changping_df['wd'].fillna(changping_df['wd'].mode()[0])

print(f"\nMissing Value:\n{changping_df.isna().sum()}")
changping_df.describe()
changping_df.sample(5)


dingling_df['TEMP'] = dingling_df['TEMP'].fillna(dingling_df['TEMP'].median())
dingling_df['PRES'] = dingling_df['PRES'].fillna(dingling_df['PRES'].median())
dingling_df['DEWP'] = dingling_df['DEWP'].fillna(dingling_df['DEWP'].median())
dingling_df['RAIN'] = dingling_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
dingling_df['WSPM'] = dingling_df['WSPM'].fillna(dingling_df['WSPM'].median())

dingling_df['datetime'] = pd.to_datetime(dingling_df[['year', 'month', 'day', 'hour']])
dingling_df.set_index('datetime', inplace=True)
dingling_df[cols_to_interpolate] = dingling_df[cols_to_interpolate].interpolate(method='time')
dingling_df['wd'] = dingling_df['wd'].fillna(dingling_df['wd'].mode()[0])

print(f"\nMissing Value:\n{dingling_df.isna().sum()}")
print(dingling_df[dingling_df['NO2'].isna()])

dingling_df['NO2'] = dingling_df['NO2'].fillna(dingling_df['NO2'].median())
print(f"\nMissing Value:\n{dingling_df.isna().sum()}")
dingling_df.describe()
dingling_df.sample(5)


dongsi_df['TEMP'] = dongsi_df['TEMP'].fillna(dongsi_df['TEMP'].median())
dongsi_df['PRES'] = dongsi_df['PRES'].fillna(dongsi_df['PRES'].median())
dongsi_df['DEWP'] = dongsi_df['DEWP'].fillna(dongsi_df['DEWP'].median())
dongsi_df['RAIN'] = dongsi_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
dongsi_df['WSPM'] = dongsi_df['WSPM'].fillna(dongsi_df['WSPM'].median())


dongsi_df['datetime'] = pd.to_datetime(dongsi_df[['year', 'month', 'day', 'hour']])
dongsi_df.set_index('datetime', inplace=True)
dongsi_df[cols_to_interpolate] = dongsi_df[cols_to_interpolate].interpolate(method='time')
dongsi_df['wd'] = dongsi_df['wd'].fillna(dongsi_df['wd'].mode()[0])

print(f"\nMissing Value:\n{dongsi_df.isna().sum()}")
dongsi_df.describe()
dongsi_df.sample(5)


guanyuan_df['TEMP'] = guanyuan_df['TEMP'].fillna(guanyuan_df['TEMP'].median())
guanyuan_df['PRES'] = guanyuan_df['PRES'].fillna(guanyuan_df['PRES'].median())
guanyuan_df['DEWP'] = guanyuan_df['DEWP'].fillna(guanyuan_df['DEWP'].median())
guanyuan_df['RAIN'] = guanyuan_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
guanyuan_df['WSPM'] = guanyuan_df['WSPM'].fillna(guanyuan_df['WSPM'].median())

guanyuan_df['datetime'] = pd.to_datetime(guanyuan_df[['year', 'month', 'day', 'hour']])
guanyuan_df.set_index('datetime', inplace=True)
guanyuan_df[cols_to_interpolate] = guanyuan_df[cols_to_interpolate].interpolate(method='time')
guanyuan_df['wd'] = guanyuan_df['wd'].fillna(guanyuan_df['wd'].mode()[0])

print(f"\nMissing Value:\n{guanyuan_df.isna().sum()}")
guanyuan_df.describe()
guanyuan_df.sample(5)


gucheng_df['TEMP'] = gucheng_df['TEMP'].fillna(gucheng_df['TEMP'].median())
gucheng_df['PRES'] = gucheng_df['PRES'].fillna(gucheng_df['PRES'].median())
gucheng_df['DEWP'] = gucheng_df['DEWP'].fillna(gucheng_df['DEWP'].median())
gucheng_df['RAIN'] = gucheng_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
gucheng_df['WSPM'] = gucheng_df['WSPM'].fillna(gucheng_df['WSPM'].median())

gucheng_df['datetime'] = pd.to_datetime(gucheng_df[['year', 'month', 'day', 'hour']])
gucheng_df.set_index('datetime', inplace=True)
gucheng_df[cols_to_interpolate] = gucheng_df[cols_to_interpolate].interpolate(method='time')
gucheng_df['wd'] = gucheng_df['wd'].fillna(gucheng_df['wd'].mode()[0])

print(f"\nMissing Value:\n{gucheng_df.isna().sum()}")
print(gucheng_df[gucheng_df['NO2'].isna()])

gucheng_df['NO2'] = gucheng_df['NO2'].fillna(gucheng_df['NO2'].median())
print(f"\nMissing Value:\n{gucheng_df.isna().sum()}")
gucheng_df.describe()
gucheng_df.sample(5)


huairou_df['TEMP'] = huairou_df['TEMP'].fillna(huairou_df['TEMP'].median())
huairou_df['PRES'] = huairou_df['PRES'].fillna(huairou_df['PRES'].median())
huairou_df['DEWP'] = huairou_df['DEWP'].fillna(huairou_df['DEWP'].median())
huairou_df['RAIN'] = huairou_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
huairou_df['WSPM'] = huairou_df['WSPM'].fillna(huairou_df['WSPM'].median())

huairou_df['datetime'] = pd.to_datetime(huairou_df[['year', 'month', 'day', 'hour']])
huairou_df.set_index('datetime', inplace=True)
huairou_df[cols_to_interpolate] = huairou_df[cols_to_interpolate].interpolate(method='time')
huairou_df['wd'] = huairou_df['wd'].fillna(huairou_df['wd'].mode()[0])

print(f"\nMissing Value:\n{huairou_df.isna().sum()}")
huairou_df.describe()
huairou_df.sample(5)


nongzhanguan_df['TEMP'] = nongzhanguan_df['TEMP'].fillna(nongzhanguan_df['TEMP'].median())
nongzhanguan_df['PRES'] = nongzhanguan_df['PRES'].fillna(nongzhanguan_df['PRES'].median())
nongzhanguan_df['DEWP'] = nongzhanguan_df['DEWP'].fillna(nongzhanguan_df['DEWP'].median())
nongzhanguan_df['RAIN'] = nongzhanguan_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
nongzhanguan_df['WSPM'] = nongzhanguan_df['WSPM'].fillna(nongzhanguan_df['WSPM'].median())

nongzhanguan_df['datetime'] = pd.to_datetime(nongzhanguan_df[['year', 'month', 'day', 'hour']])
nongzhanguan_df.set_index('datetime', inplace=True)
nongzhanguan_df[cols_to_interpolate] = nongzhanguan_df[cols_to_interpolate].interpolate(method='time')
nongzhanguan_df['wd'] = nongzhanguan_df['wd'].fillna(nongzhanguan_df['wd'].mode()[0])

print(f"\nMissing Value:\n{nongzhanguan_df.isna().sum()}")
nongzhanguan_df.describe()
nongzhanguan_df.sample(5)


shunyi_df['TEMP'] = shunyi_df['TEMP'].fillna(shunyi_df['TEMP'].median())
shunyi_df['PRES'] = shunyi_df['PRES'].fillna(shunyi_df['PRES'].median())
shunyi_df['DEWP'] = shunyi_df['DEWP'].fillna(shunyi_df['DEWP'].median())
shunyi_df['RAIN'] = shunyi_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
shunyi_df['WSPM'] = shunyi_df['WSPM'].fillna(shunyi_df['WSPM'].median())

shunyi_df['datetime'] = pd.to_datetime(shunyi_df[['year', 'month', 'day', 'hour']])
shunyi_df.set_index('datetime', inplace=True)
shunyi_df[cols_to_interpolate] = shunyi_df[cols_to_interpolate].interpolate(method='time')
shunyi_df['wd'] = shunyi_df['wd'].fillna(shunyi_df['wd'].mode()[0])

print(f"\nMissing Value:\n{shunyi_df.isna().sum()}")
shunyi_df.describe()
shunyi_df.sample(5)


tiantian_df['TEMP'] = tiantian_df['TEMP'].fillna(tiantian_df['TEMP'].median())
tiantian_df['PRES'] = tiantian_df['PRES'].fillna(tiantian_df['PRES'].median())
tiantian_df['DEWP'] = tiantian_df['DEWP'].fillna(tiantian_df['DEWP'].median())
tiantian_df['RAIN'] = tiantian_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
tiantian_df['WSPM'] = tiantian_df['WSPM'].fillna(tiantian_df['WSPM'].median())

tiantian_df['datetime'] = pd.to_datetime(tiantian_df[['year', 'month', 'day', 'hour']])
tiantian_df.set_index('datetime', inplace=True)
tiantian_df[cols_to_interpolate] = tiantian_df[cols_to_interpolate].interpolate(method='time')
tiantian_df['wd'] = tiantian_df['wd'].fillna(tiantian_df['wd'].mode()[0])

print(f"\nMissing Value:\n{tiantian_df.isna().sum()}")
tiantian_df.describe()
tiantian_df.sample(5)


wanliu_df['TEMP'] = wanliu_df['TEMP'].fillna(wanliu_df['TEMP'].median())
wanliu_df['PRES'] = wanliu_df['PRES'].fillna(wanliu_df['PRES'].median())
wanliu_df['DEWP'] = wanliu_df['DEWP'].fillna(wanliu_df['DEWP'].median())
wanliu_df['RAIN'] = wanliu_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
wanliu_df['WSPM'] = wanliu_df['WSPM'].fillna(wanliu_df['WSPM'].median())

wanliu_df['datetime'] = pd.to_datetime(wanliu_df[['year', 'month', 'day', 'hour']])
wanliu_df.set_index('datetime', inplace=True)
wanliu_df[cols_to_interpolate] = wanliu_df[cols_to_interpolate].interpolate(method='time')
wanliu_df['wd'] = wanliu_df['wd'].fillna(wanliu_df['wd'].mode()[0])

print(f"\nMissing Value:\n{wanliu_df.isna().sum()}")
wanliu_df.describe()
wanliu_df.sample(5)


wanshouxigong_df['TEMP'] = wanshouxigong_df['TEMP'].fillna(wanshouxigong_df['TEMP'].median())
wanshouxigong_df['PRES'] = wanshouxigong_df['PRES'].fillna(wanshouxigong_df['PRES'].median())
wanshouxigong_df['DEWP'] = wanshouxigong_df['DEWP'].fillna(wanshouxigong_df['DEWP'].median())
wanshouxigong_df['RAIN'] = wanshouxigong_df['RAIN'].fillna(0)  # Karena hujan seringkali 0
wanshouxigong_df['WSPM'] = wanshouxigong_df['WSPM'].fillna(wanshouxigong_df['WSPM'].median())

wanshouxigong_df['datetime'] = pd.to_datetime(wanshouxigong_df[['year', 'month', 'day', 'hour']])
wanshouxigong_df.set_index('datetime', inplace=True)
wanshouxigong_df[cols_to_interpolate] = wanshouxigong_df[cols_to_interpolate].interpolate(method='time')
wanshouxigong_df['wd'] = wanshouxigong_df['wd'].fillna(wanshouxigong_df['wd'].mode()[0])

print(f"\nMissing Value:\n{wanshouxigong_df.isna().sum()}")
wanshouxigong_df.describe()
wanshouxigong_df.sample(5)


df_dict = {
    "Aotizhongxin": aotizhongxin_df,
    "Changping": changping_df,
    "Dingling": dingling_df,
    "Dongsi": dongsi_df,
    "Guanyuan": guanyuan_df,
    "Gucheng": gucheng_df,
    "Huairou": huairou_df,
    "Nongzhanguan": nongzhanguan_df,
    "Shunyi": shunyi_df,
    "Tiantian": tiantian_df,
    "Wanliu": wanliu_df,
    "Wanshouxigong": wanshouxigong_df
}


for nama, df in df_dict.items():
    print(f"Struktur Data Stasiun {nama}:\n")
    display(df.info())

for nama, df in df_dict.items():
    print(f"Statistik Data Stasiun {nama}:\n")
    display(df.describe())

for nama, df in df_dict.items():
    plt.figure(figsize=(8, 4))
    sns.histplot(df["PM2.5"].dropna(), bins=30, kde=True)
    plt.title(f"Distribusi PM2.5 di Stasiun {nama}")
    plt.xlabel("PM2.5")
    plt.ylabel("Frekuensi")
    plt.show()

#IMPLEMENTASI STREAMLIT
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
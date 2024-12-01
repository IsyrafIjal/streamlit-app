import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache
def load_data():
    file_path = 'Depression Professional Dataset.csv'  # Sesuaikan dengan path
    return pd.read_csv(file_path)

df = load_data()

# Judul Aplikasi
st.title("Analisis Dataset Profesional dengan Depresi")

# Menampilkan data
st.header("Tinjauan Dataset")
st.write("### Dataframe:")
st.dataframe(df.head())
st.write("### Statistik Deskriptif:")
st.write(df.describe())

# Visualisasi: Distribusi Usia
st.header("Distribusi Usia")
fig, ax = plt.subplots()
sns.histplot(df['Age'], kde=True, ax=ax)
st.pyplot(fig)

# Visualisasi: Gender dan Depresi
st.header("Distribusi Depresi berdasarkan Gender")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Depression", hue="Gender", ax=ax)
st.pyplot(fig)

# Analisis Korelasi
st.header("Analisis Korelasi")
corr = df.select_dtypes(include=['float64', 'int64']).corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Analisis berdasarkan kolom pilihan
st.header("Analisis berdasarkan Kolom Pilihan")
columns = df.columns.tolist()
columns.remove("Depression")  # Menghapus kolom target dari opsi

selected_columns = st.multiselect("Pilih kolom untuk analisis:", columns, default=["Gender"])
if selected_columns:
    st.write(f"### Analisis untuk kolom: {', '.join(selected_columns)}")
    for col in selected_columns:
        st.write(f"#### Distribusi berdasarkan kolom: {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, hue="Depression", ax=ax)
        ax.set_title(f"Distribusi Depression untuk {col}")
        st.pyplot(fig)
else:
    st.write("Pilih setidaknya satu kolom untuk analisis!")

# Analisis kombinasi kolom
st.header("Analisis Kombinasi Kolom dengan Depresi Tertinggi")
comb_columns = st.multiselect("Pilih kolom untuk kombinasi:", columns, default=["Gender", "Family History of Mental Illness", "Work Pressure"])
if comb_columns:
    st.write(f"### Kombinasi kolom: {', '.join(comb_columns)}")
    combination_df = df.groupby(comb_columns)["Depression"].value_counts(normalize=True).unstack().fillna(0)
    combination_df = combination_df.sort_values(by="Yes", ascending=False)
    st.write("#### Kombinasi dengan proporsi tertinggi:")
    st.dataframe(combination_df.head(10))
else:
    st.write("Pilih setidaknya satu kolom untuk analisis kombinasi!")

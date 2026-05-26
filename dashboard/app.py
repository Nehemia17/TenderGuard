import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys

# Tambahkan root folder ke PYTHONPATH
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

try:
    from modules.predictor import load_classifier, predict_text, get_feature_importances
    from modules.feature_extractor import load_vectorizer
    from modules.preprocessor import clean_text
    MODEL_READY = True
except ImportError as e:
    MODEL_READY = False
    st.error(f"Error import modul internal: {e}")

# ══════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="HoaxGuard | Advanced Data Mining",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Helper Scraper (Basic)
def scrape_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text if len(text) > 50 else "Gagal mengambil teks yang cukup panjang dari URL."
    except Exception as e:
        return f"Error scraping URL: {e}"

# ══════════════════════════════════════════════
# TABS LOGIC
# ══════════════════════════════════════════════

def tab_inference(vectorizer, classifier):
    st.markdown("<div class='section-title'><span>🔍 Mesin Analisis</span> Teks & URL</div>", unsafe_allow_html=True)
    
    col_input, col_info = st.columns([7, 3], gap="large")
    
    with col_input:
        input_mode = st.radio("Mode Input:", ["Teks Manual", "Ekstraksi URL"], horizontal=True)
        
        if input_mode == "Teks Manual":
            input_text = st.text_area(
                "input_berita", 
                label_visibility="collapsed",
                height=220, 
                placeholder="Ketik atau paste pesan berantai di sini...\n\nContoh:\n\"Vaksin mengandung chip mikro elit global!\""
            )
        else:
            input_url = st.text_input("Masukkan URL Artikel Berita:", placeholder="https://news.com/artikel...")
            input_text = ""
            if input_url:
                with st.spinner("Scraping konten..."):
                    input_text = scrape_url(input_url)
                    st.text_area("Teks Terekstrak:", value=input_text[:1000] + "...", height=150, disabled=True)
        
        analyze_btn = st.button("🚀 Analisis Teks dengan AI", use_container_width=True)
        
    with col_info:
        st.markdown("<div class='code-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#faff69; margin-top:0;'>💡 Ekstraksi Fitur Lanjut</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#ccc; line-height:1.5;'>Sistem memecah teks menjadi <i>TF-IDF vectors</i> dan mengekstrak metrik gaya bahasa. Fitur ini meniru pemikiran analitik (CRISP-DM Modeling Phase).</p>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#2a2a2a;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Didukung Algoritma:</p>", unsafe_allow_html=True)
        st.markdown("<ul style='font-size:0.85rem; color:#ccc; padding-left:1.2rem;'>"
                    "<li>Logistic Regression (Main)</li>"
                    "<li>TF-IDF N-Gram</li>"
                    "<li>Sastrawi Stemming</li></ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn:
        if not input_text.strip() or input_text.startswith("Error"):
            st.error("⚠️ Input tidak valid atau kosong!")
            return
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Menjalankan Model Machine Learning..."):
            time.sleep(0.8) # Simulasi komputasi berat
            result = predict_text(input_text, vectorizer, classifier, include_extra=False)
            
            st.markdown("<div class='section-title'><span>📈 Hasil</span> Keputusan AI</div>", unsafe_allow_html=True)
            
            col_res_left, col_res_right = st.columns([1, 2], gap="large")
            
            with col_res_left:
                label = result["label"]
                conf = result["confidence"]
                
                if label == "HOAKS":
                    st.markdown(f"""
                    <div style="background-color:rgba(239,68,68,0.1); border:1px solid #ef4444; padding:2rem; border-radius:12px; text-align:center;">
                        <h2 style="color:#ef4444; margin:0;">🚨 {label}</h2>
                        <p style="color:#ccc; margin-top:0.5rem;">Kepastian Model: <b>{conf:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:rgba(34,197,94,0.1); border:1px solid #22c55e; padding:2rem; border-radius:12px; text-align:center;">
                        <h2 style="color:#22c55e; margin:0;">✅ {label}</h2>
                        <p style="color:#ccc; margin-top:0.5rem;">Kepastian Model: <b>{conf:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_res_right:
                st.markdown("<div class='code-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0; color:#fff;'>Bedah Linguistik (Explainable AI)</h4>", unsafe_allow_html=True)
                st.markdown(f"**Teks Asli (Truncated):**\n> *{input_text[:150]}...*")
                st.markdown(f"**Token Bersih (Sastrawi Stemmer):**\n`<span style='color:#faff69;'>{result['cleaned_text'][:200]}...</span>`", unsafe_allow_html=True)
                
                if label == "HOAKS":
                    importances = get_feature_importances(classifier, vectorizer, top_n=50)
                    cleaned_words = result["cleaned_text"].split()
                    red_flags = [word for word, score in importances if word in cleaned_words]
                    
                    if red_flags:
                        badges = " ".join([f"<span class='badge-high'>{w}</span>" for w in red_flags])
                        st.markdown(f"<br>**Kata Pemicu Hoaks (Red Flags):**<br>{badges}", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

def tab_eda():
    st.markdown("<div class='section-title'><span>📊 Exploratory Data Analysis</span> (CRISP-DM: Data Understanding)</div>", unsafe_allow_html=True)
    
    data_path = root_path / "data" / "processed" / "dataset_hoax_demo.csv"
    if not data_path.exists():
        st.info("Dataset tidak ditemukan. Jalankan generate_demo_data.py")
        return
        
    df = pd.read_csv(data_path)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel Berita", f"{len(df):,}")
    col2.metric("Berita Valid", f"{len(df[df['label']==0]):,}")
    col3.metric("Berita Hoaks", f"{len(df[df['label']==1]):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("#### Keseimbangan Kelas (Class Balance)")
        try:
            import plotly.express as px
            fig = px.pie(
                names=["Valid", "Hoaks"],
                values=[len(df[df['label']==0]), len(df[df['label']==1])],
                hole=0.5,
                color=["Valid", "Hoaks"],
                color_discrete_map={"Valid":"#22c55e", "Hoaks":"#ef4444"},
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.bar_chart(df["label"].value_counts())
            
    with c_right:
        st.markdown("#### Panjang Karakter vs Label")
        df["panjang"] = df["text"].apply(len)
        try:
            import plotly.express as px
            fig = px.box(
                df, x="label", y="panjang",
                color="label",
                color_discrete_map={0:"#22c55e", 1:"#ef4444"},
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.dataframe(df.groupby("label")["panjang"].describe())

def tab_evaluation():
    st.markdown("<div class='section-title'><span>📈 Model Evaluation</span> (CRISP-DM: Evaluation Phase)</div>", unsafe_allow_html=True)
    st.markdown("Membandingkan performa 3 algoritma Machine Learning berbeda pada dataset uji (Test Set 20%).")
    
    metrics_path = root_path / "model" / "evaluation_metrics.json"
    if not metrics_path.exists():
        st.warning("File evaluasi tidak ditemukan. Harap jalankan script training.")
        return
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    # Buat tabel perbandingan
    comp_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Akurasi": metrics["LogisticRegression"]["accuracy"], "F1-Score": metrics["LogisticRegression"]["f1"]},
        {"Model": "Naive Bayes", "Akurasi": metrics["NaiveBayes"]["accuracy"], "F1-Score": metrics["NaiveBayes"]["f1"]},
        {"Model": "Random Forest", "Akurasi": metrics["RandomForest"]["accuracy"], "F1-Score": metrics["RandomForest"]["f1"]},
    ])
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Metrik Perbandingan Algoritma")
        def format_pct(val): return f"{val:.2%}"
        st.dataframe(comp_df.style.format({"Akurasi": format_pct, "F1-Score": format_pct}).background_gradient(cmap="YlOrRd"), use_container_width=True)
        
    with c2:
        st.markdown("#### Confusion Matrix (Logistic Regression)")
        cm = metrics["LogisticRegression"]["cm"]
        try:
            import plotly.figure_factory as ff
            z = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
            x = ['Prediksi Valid', 'Prediksi Hoaks']
            y = ['Aktual Valid', 'Aktual Hoaks']
            fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Reds')
            fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", margin=dict(t=20, b=20, l=20, r=20), font=dict(color="#fff"))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.write(cm)

# ══════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════
def main():
    load_css()
    
    st.markdown("""
    <div class="tg-header">
        <h1><span class="brand">Hoax</span>Guard</h1>
        <p>Advanced Disinformation Detection Engine &bull; GEMASTIK XVIII Data Mining Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODEL_READY:
        return
        
    try:
        vectorizer = load_vectorizer()
        classifier = load_classifier()
    except FileNotFoundError:
        st.warning("⚠️ **Model Belum Dilatih!** Jalankan `python generate_demo_data.py` terlebih dahulu.")
        return
        
    # Tab Navigation
    t1, t2, t3 = st.tabs(["🚀 Deteksi Teks", "📊 Analisis Dataset (EDA)", "📈 Evaluasi Model"])
    
    with t1:
        tab_inference(vectorizer, classifier)
    with t2:
        tab_eda()
    with t3:
        tab_evaluation()

if __name__ == "__main__":
    main()

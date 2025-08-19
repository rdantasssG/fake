# app.py
import streamlit as st
import logging
import traceback
from pags.fakeD import AdvancedVideoAnalyzer

# Configuração do logger
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Detector de Vídeos IA", page_icon="🎥", layout="centered")

st.title("🎥 Detector de Vídeos Sintéticos (IA)")

# Escolha do tipo de entrada
option = st.radio("Como deseja analisar o vídeo?", ["YouTube URL", "Upload de Arquivo"])

youtube_url = None
uploaded_file = None

if option == "YouTube URL":
    youtube_url = st.text_input("Cole aqui o link do vídeo do YouTube:")
else:
    uploaded_file = st.file_uploader("Faça upload do vídeo", type=["mp4", "avi", "mov"])

if st.button("Analisar Vídeo"):
    try:
        analyzer = AdvancedVideoAnalyzer(confidence_threshold=0.65)

        with st.spinner("⏳ Analisando o vídeo... isso pode levar alguns minutos."):
            if youtube_url:
                result = analyzer.analyze_video_from_url(youtube_url)
            elif uploaded_file:
                temp_path = "uploaded_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                result = analyzer.analyze_video_file(temp_path)
            else:
                st.warning("⚠️ Insira uma URL ou faça upload de um vídeo.")
                result = None

        if result is None:
            st.error("❌ Ocorreu um erro durante a análise.")
        else:
            st.success("✅ Análise concluída!")

            # Exibir resumo
            st.subheader("📊 Resultados da Análise")
            st.write(f"**Título:** {result.details.get('youtube_info', {}).get('title', 'N/A')}")
            st.write(f"**Canal:** {result.details.get('youtube_info', {}).get('uploader', 'N/A')}")

            st.metric("Artefatos Visuais", f"{result.visual_artifacts_score:.2f}")
            st.metric("Consistência Temporal", f"{result.temporal_consistency_score:.2f}")
            st.metric("Análise de Frequência", f"{result.frequency_analysis_score:.2f}")
            st.metric("Consistência de Bordas", f"{result.edge_consistency_score:.2f}")
            st.metric("Padrões de Movimento", f"{result.motion_analysis_score:.2f}")

            prob = result.final_score * 100
            if result.is_synthetic:
                st.error(f"🚨 PROBABILIDADE DE SER SINTÉTICO: {prob:.2f}% (Acima do limiar)")
            else:
                st.success(f"✅ PROBABILIDADE DE SER AUTÊNTICO: {100-prob:.2f}% (Abaixo do limiar)")

    except Exception as e:
        st.error("⚠️ Erro inesperado!")
        st.code(str(e))
        st.text_area("Detalhes técnicos", traceback.format_exc())

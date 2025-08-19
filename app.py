# app.py
import streamlit as st
import logging
from pags.fakeD import AdvancedVideoAnalyzer  # 👈 ajuste: importe do arquivo onde sua classe está definida

# Configuração do logger (opcional, para debug no terminal)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Detector de Vídeos IA", page_icon="🎥", layout="centered")

st.title("🎥 Detector de Vídeos Sintéticos (IA)")

# Input do usuário
youtube_url = st.text_input("Cole aqui o link do vídeo do YouTube:")

if st.button("Analisar Vídeo"):
    if not youtube_url:
        st.warning("⚠️ Insira uma URL válida do YouTube antes de analisar.")
    else:
        analyzer = AdvancedVideoAnalyzer(confidence_threshold=0.65)

        with st.spinner("⏳ Analisando o vídeo... isso pode levar alguns minutos."):
            result = analyzer.analyze_video_from_url(youtube_url)

        if result is None:
            st.error("❌ Ocorreu um erro durante a análise. Verifique a URL ou tente novamente.")
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

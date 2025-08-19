# -*- coding: utf-8 -*-

# Requisitos de instalação:
import cv2
import numpy as np
import os
import logging
import tempfile
import shutil
import streamlit as st
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from scipy.fft import fft2, fftshift
import json
from datetime import datetime
import re
import subprocess

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Estrutura para armazenar resultados da análise"""
    visual_artifacts_score: float
    temporal_consistency_score: float
    frequency_analysis_score: float
    edge_consistency_score: float
    motion_analysis_score: float
    final_score: float
    is_synthetic: bool
    confidence: float
    details: Dict

class YouTubeVideoDownloader:
    """
    Classe para download de vídeos do YouTube usando yt-dlp de forma robusta.
    """

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ai_detector_")
        self.downloaded_files = []

    def __del__(self):
        """Limpa arquivos temporários ao destruir o objeto"""
        self.cleanup()

    def cleanup(self):
        """Remove o diretório e os arquivos temporários"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Arquivos temporários removidos: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Erro ao limpar arquivos temporários: {e}")

    def is_valid_youtube_url(self, url: str) -> bool:
        """Verifica se a URL é válida do YouTube usando regex"""
        youtube_regex = re.compile(
            r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)'
            r'([a-zA-Z0-9_-]{11})'
        )
        return bool(youtube_regex.match(url))

    def extract_video_id(self, url: str) -> str:
        """Extrai o ID do vídeo da URL do YouTube"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError("Não foi possível extrair o ID do vídeo da URL")

    def check_yt_dlp_installation(self) -> bool:
        """Verifica se o yt-dlp está instalado e acessível no PATH"""
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_video_info(self, url: str) -> Dict:
        """Obtém metadados do vídeo (título, duração, etc.) sem baixá-lo"""
        try:
            cmd = ['yt-dlp', '--dump-json', '--no-download', url]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True, timeout=30)
            video_info = json.loads(result.stdout)
            return {
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'uploader': video_info.get('uploader', 'Unknown')
            }
        except Exception as e:
            logger.warning(f"Não foi possível obter informações do vídeo: {e}")
            return {}

    def download_video(self, url: str, quality: str = 'best[height<=720]') -> str:
        """Baixa o vídeo do YouTube usando yt-dlp"""
        if not self.is_valid_youtube_url(url):
            raise ValueError("URL do YouTube inválida")

        if not self.check_yt_dlp_installation():
            raise RuntimeError("yt-dlp não está instalado ou não está no PATH. Instale com: pip install yt-dlp")

        video_id = self.extract_video_id(url)
        output_template = os.path.join(self.temp_dir, f"video_{video_id}.%(ext)s")

        cmd = ['yt-dlp', '-f', quality, '-o', output_template, '--no-playlist', '--no-warnings', '--socket-timeout', '30', url]

        logger.info(f"Iniciando download do vídeo: {url} com qualidade: {quality}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)

            for file in os.listdir(self.temp_dir):
                if file.startswith(f"video_{video_id}"):
                    downloaded_file = os.path.join(self.temp_dir, file)
                    self.downloaded_files.append(downloaded_file)
                    logger.info(f"Download concluído: {downloaded_file}")
                    return downloaded_file

            raise FileNotFoundError("Arquivo baixado não encontrado no diretório temporário")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout no download do vídeo (>5 minutos)")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            raise RuntimeError(f"Erro no download via yt-dlp: {error_msg}")

class AdvancedVideoAnalyzer:
    """
    Analisador de vídeo para detecção de conteúdo sintético/IA via heurísticas.
    """

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.downloader = YouTubeVideoDownloader()

    def extract_frames_smart(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        """Extração de frames com distribuição uniforme ao longo do vídeo"""
        frames = []
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Erro: Não foi possível abrir o vídeo: {video_path}")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            video_capture.release()
            raise ValueError("Vídeo não contém frames ou está corrompido.")

        frame_interval = max(1, total_frames // max_frames)

        frame_count = 0
        extracted_count = 0
        while extracted_count < max_frames:
            # Pula para o frame desejado
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = video_capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame_resized)
            extracted_count += 1
            frame_count += frame_interval

        video_capture.release()
        logger.info(f"Extraídos {len(frames)} frames para análise")
        return frames

    def analyze_visual_artifacts(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Análise de artefatos visuais com calibração para amplificação"""
        artifact_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))

            blur_score = min(1.0, max(0.0, (600 - blur_variance) / 600)) if blur_variance < 600 else 0
            noise_score = min(1.0, noise_level / 50.0)

            frame_score = (blur_score * 0.5 + noise_score * 0.5) ** 0.75
            artifact_scores.append(frame_score)

        avg_score = np.mean(artifact_scores) if artifact_scores else 0.0
        score_variance = np.var(artifact_scores) if artifact_scores else 0.0
        return avg_score, {'score_variance': score_variance}

    def analyze_temporal_consistency(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Análise de consistência temporal com calibração para amplificação"""
        if len(frames) < 3: return 0.0, {}
        consistency_scores = []
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        prev_points = cv2.goodFeaturesToTrack(gray_frames[0], maxCorners=150, qualityLevel=0.01, minDistance=10)

        for i in range(1, len(gray_frames)):
            if prev_points is None or len(prev_points) < 5:
                prev_points = cv2.goodFeaturesToTrack(gray_frames[i-1], maxCorners=150, qualityLevel=0.01, minDistance=10)
                if prev_points is None: continue

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_frames[i-1], gray_frames[i], prev_points, None)

            if next_points is not None and status.any():
                good_new = next_points[status.flatten() == 1]
                good_old = prev_points[status.flatten() == 1]

                if len(good_new) > 5:
                    motion_variance = np.std(np.linalg.norm(good_new - good_old, axis=1))
                    motion_score = min(1.0, motion_variance / 25.0)
                else:
                    motion_score = 0.0

                frame_diff_mean = np.mean(cv2.absdiff(gray_frames[i-1], gray_frames[i]))
                diff_score = min(1.0, frame_diff_mean / 75.0)

                consistency_scores.append(motion_score * 0.6 + diff_score * 0.4)
                prev_points = good_new.reshape(-1, 1, 2)
            else:
                prev_points = None

        avg_score = np.mean(consistency_scores) if consistency_scores else 0.0
        return avg_score ** 0.8, {}

    def analyze_frequency_domain(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Análise no domínio da frequência para detectar padrões artificiais"""
        if not frames: return 0.0, {}
        frequency_scores = []
        for frame in frames[:min(20, len(frames))]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            f_transform = fft2(gray)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)

            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            mid_freq_energy = np.mean(magnitude_spectrum[(radius >= 20) & (radius < 60)])
            high_freq_energy = np.mean(magnitude_spectrum[radius >= 60])
            energy_ratio = high_freq_energy / (mid_freq_energy + 1e-10)

            freq_score = min(1.0, max(0.0, (energy_ratio - 1.0) * 2.0))
            frequency_scores.append(freq_score)
        return np.mean(frequency_scores) if frequency_scores else 0.0, {}

    def analyze_edge_consistency(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Análise de consistência de bordas com calibração para amplificação"""
        if len(frames) < 2: return 0.0, {}
        edge_scores = []
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        prev_edges = cv2.Canny(gray_frames[0], 50, 150)
        for i in range(1, len(gray_frames)):
            edges = cv2.Canny(gray_frames[i], 50, 150)
            edge_change = np.mean(cv2.absdiff(edges, prev_edges)) / 255.0
            edge_score = min(1.0, edge_change * 10.0)
            edge_scores.append(edge_score)
            prev_edges = edges
        avg_score = np.mean(edge_scores) if edge_scores else 0.0
        return avg_score ** 0.8, {}

    def analyze_motion_patterns(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Análise de padrões de movimento com calibração para amplificação"""
        if len(frames) < 5: return 0.0, {}
        motion_scores = []
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        orb = cv2.ORB_create(nfeatures=500)
        for i in range(len(gray_frames) - 1):
            kp1, des1 = orb.detectAndCompute(gray_frames[i], None)
            kp2, des2 = orb.detectAndCompute(gray_frames[i + 1], None)
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
                    motion_variance = np.var(np.linalg.norm(dst_pts - src_pts, axis=1))
                    motion_score = min(1.0, motion_variance / 800.0)
                    motion_scores.append(motion_score)
        avg_score = np.mean(motion_scores) if motion_scores else 0.0
        return avg_score ** 0.8, {}

    def analyze_video_from_url(self, youtube_url: str, quality: str = 'best[height<=720]') -> Optional[AnalysisResult]:
        """Orquestra a análise de um vídeo a partir de uma URL do YouTube"""
        logger.info(f"Iniciando análise do vídeo do YouTube: {youtube_url}")
        try:
            video_info = self.downloader.get_video_info(youtube_url)
            video_path = self.downloader.download_video(youtube_url, quality)
            result = self.analyze_video(video_path)
            result.details['youtube_info'] = video_info
            result.details['youtube_url'] = youtube_url
            self._print_results(result)
            self.save_detailed_report(result)
            return result
        except Exception as e:
            logger.error(f"Erro fatal na análise do vídeo do YouTube: {e}")
            return None

    def analyze_video(self, video_path: str) -> AnalysisResult:
        """Executa a suíte de análise completa em um arquivo de vídeo local, com instrumentação"""
        logger.info(f"Iniciando análise completa do vídeo: {video_path}")
        frames = self.extract_frames_smart(video_path, max_frames=50)
        if len(frames) < 5: raise ValueError("Vídeo muito curto para análise confiável")

        visual_score, visual_details = self.analyze_visual_artifacts(frames)
        temporal_score, _ = self.analyze_temporal_consistency(frames)
        frequency_score, _ = self.analyze_frequency_domain(frames)
        edge_score, _ = self.analyze_edge_consistency(frames)
        motion_score, _ = self.analyze_motion_patterns(frames)

        # --- INSTRUMENTAÇÃO PARA DEPURAÇÃO ---
        print("\n--- SCORES INDIVIDUAIS (DEBUG) ---")
        print(f"  Visual Score:       {visual_score:.4f}")
        print(f"  Temporal Score:     {temporal_score:.4f}")
        print(f"  Frequency Score:    {frequency_score:.4f}")
        print(f"  Edge Score:         {edge_score:.4f}")
        print(f"  Motion Score:       {motion_score:.4f}")
        print("------------------------------------\n")

        weights = {'visual': 0.25, 'temporal': 0.30, 'frequency': 0.15, 'edge': 0.15, 'motion': 0.15}
        final_score = (visual_score * weights['visual'] + temporal_score * weights['temporal'] +
                       frequency_score * weights['frequency'] + edge_score * weights['edge'] +
                       motion_score * weights['motion'])

        is_synthetic = final_score > self.confidence_threshold

        return AnalysisResult(
            visual_artifacts_score=visual_score, temporal_consistency_score=temporal_score,
            frequency_analysis_score=frequency_score, edge_consistency_score=edge_score,
            motion_analysis_score=motion_score, final_score=final_score, is_synthetic=is_synthetic,
            confidence=final_score, details={'video_path': video_path, **visual_details}
        )

    def _print_results(self, result: AnalysisResult):
        """Imprime os resultados da análise de forma clara e formatada"""
        print("="*60)
        print("              RESULTADO DA ANÁLISE DE DETECÇÃO DE IA")
        print("="*60)
        if 'youtube_info' in result.details:
            info = result.details.get('youtube_info', {})
            print(f" Título: {info.get('title', 'N/A')}")
            print(f" Canal: {info.get('uploader', 'N/A')}")
        print("-"*60)
        print(f" Artefatos Visuais:         {result.visual_artifacts_score:.4f}")
        print(f" Consistência Temporal:     {result.temporal_consistency_score:.4f}")
        print(f" Análise de Frequência:     {result.frequency_analysis_score:.4f}")
        print(f" Consistência de Bordas:    {result.edge_consistency_score:.4f}")
        print(f" Padrões de Movimento:      {result.motion_analysis_score:.4f}")
        print("-"*60)
        probabilidade_sintetico = result.final_score * 100
        print(f" PROBABILIDADE DE SER SINTÉTICO: {probabilidade_sintetico:.2f}%")
        print("-"*60)
        if result.is_synthetic:
            print(f" 🚨 VEREDITO: ALTA PROBABILIDADE (Acima do limiar de {self.confidence_threshold*100:.0f}%)")
        else:
            print(f" ✅ VEREDITO: BAIXA PROBABILIDADE (Abaixo do limiar de {self.confidence_threshold*100:.0f}%)")
        print("="*60 + "\n")

    def save_detailed_report(self, result: AnalysisResult, output_dir: str = "reports"):
        """Salva um relatório detalhado da análise em formato JSON"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = "local_file"
        if 'youtube_url' in result.details:
            try:
                video_id = self.downloader.extract_video_id(result.details['youtube_url'])
            except ValueError:
                video_id = "youtube_invalid_id"

        report_filename = f"report_{video_id}_{timestamp}.json"
        report_path = os.path.join(output_dir, report_filename)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=4, default=str)
            logger.info(f"Relatório detalhado salvo em: {report_path}")
        except Exception as e:
            logger.error(f"Falha ao salvar relatório JSON: {e}")

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    # --- MODIFIQUE A URL AQUI ---
    # Vídeo real de alta qualidade para teste de falsos positivos
    YOUTUBE_URL_TO_ANALYZE = "https://www.youtube.com/watch?v=olUhMl7SOfA&pp=ygUMdmlkZW8gY29tIGlh" # Exemplo: Vídeo 4K da NASA

    # Vídeo de animação para observar como as heurísticas reagem
    # YOUTUBE_URL_TO_ANALYZE = "https://www.youtube.com/watch?v=1-n6plwTqto"

    analyzer = AdvancedVideoAnalyzer(confidence_threshold=0.65)

    try:
        analyzer.analyze_video_from_url(YOUTUBE_URL_TO_ANALYZE)
    except Exception as e:
        logger.critical(f"Ocorreu um erro fatal durante a execução: {e}", exc_info=True)
    finally:
        # A limpeza dos arquivos temporários é gerenciada pelo __del__ do downloader
        logger.info("Processo de análise concluído.")

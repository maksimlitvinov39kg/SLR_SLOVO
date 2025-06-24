import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
import gdown
import warnings
import tempfile
import time
from PIL import Image
import io
import base64
from model import SOTASignLanguageModel
from preprocessor import PreprocessLayerBothHands

warnings.filterwarnings('ignore')

USE_TYPES = ['left_hand', 'pose', 'right_hand']
START_IDX = 468
LIPS_IDXS0 = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308,
])
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

N_ROWS = 543
LANDMARK_IDXS_BOTH_HANDS = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0))

LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, RIGHT_HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LEFT_POSE_IDXS0)).squeeze()

N_COLS = LANDMARK_IDXS_BOTH_HANDS.size

class KeypointsExtractor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_from_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        keypoints_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = self.holistic.process(image)
            
            frame_keypoints = self._extract_frame_keypoints(results, frame_idx)
            keypoints_data.extend(frame_keypoints)
            
            frame_idx += 1
        
        cap.release()
        
        df = pd.DataFrame(keypoints_data)
        
        if output_path:
            df.to_parquet(output_path, index=False)
        
        return df
    
    def extract_from_frames(self, frames):
        keypoints_data = []
        
        for frame_idx, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                image = frame
            
            image.flags.writeable = False
            
            results = self.holistic.process(image)
            
            frame_keypoints = self._extract_frame_keypoints(results, frame_idx)
            keypoints_data.extend(frame_keypoints)
        
        return pd.DataFrame(keypoints_data)
    
    def _extract_frame_keypoints(self, results, frame_idx):
        keypoints = []
        
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_face_{idx}',
                    'type': 'face',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
        
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_left_hand_{idx}',
                    'type': 'left_hand',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
        else:
            for idx in range(21):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_left_hand_{idx}',
                    'type': 'left_hand',
                    'landmark_index': idx,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan
                })
        
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_right_hand_{idx}',
                    'type': 'right_hand',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
        else:
            for idx in range(21):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_right_hand_{idx}',
                    'type': 'right_hand',
                    'landmark_index': idx,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan
                })
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints.append({
                    'frame': frame_idx,
                    'row_id': f'{frame_idx}_pose_{idx}',
                    'type': 'pose',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
        
        return keypoints


def load_relevant_data_subset(df):
    if df is None or len(df) == 0:
        st.error("❌ Пустой DataFrame")
        return None
    
    data_columns = ['x', 'y', 'z']
    
    missing_cols = [col for col in data_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Отсутствуют колонки: {missing_cols}")
        return None
    
    data = df[data_columns].copy()
    data.loc[data.x.isnull(), ('x')] = 0
    data.loc[data.y.isnull(), ('y')] = 0
    data.loc[data.z.isnull(), ('z')] = 0
    
    if len(data) % N_ROWS != 0:
        n_complete_frames = len(data) // N_ROWS
        data = data[:n_complete_frames * N_ROWS]
        if len(data) == 0:
            st.error("❌ Недостаточно данных для анализа")
            return None
    
    n_frames = int(len(data) / N_ROWS)
    
    data = data.values.reshape(n_frames, N_ROWS, len(data_columns))
    return data.astype(np.float32)


@st.cache_resource
def download_model_weights():
    file_id = "1lYCjuyxT8t3lebfIgO4sc1NyPe-WFZUs"
    output_path = "model_weights.pth"
    
    if os.path.exists(output_path):
        return output_path
    
    url = f'https://drive.google.com/uc?id={file_id}'
    
    with st.spinner("Скачиваем веса модели..."):
        try:
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path):
                return output_path
            else:
                return None
        except Exception as e:
            return None


@st.cache_resource
def load_model():
    try:
        if not os.path.exists('statistics.json'):
            st.error("❌ Файл statistics.json не найден!")
            return None, None, None, None
        
        with open('statistics.json', 'r') as f:
            stats = json.load(f)
        
        lips_mean = torch.tensor(stats['LIPS_MEAN'], dtype=torch.float32)
        lips_std = torch.tensor(stats['LIPS_STD'], dtype=torch.float32)
        left_hands_mean = torch.tensor(stats['LEFT_HANDS_MEAN'], dtype=torch.float32)
        left_hands_std = torch.tensor(stats['LEFT_HANDS_STD'], dtype=torch.float32)
        right_hands_mean = torch.tensor(stats['RIGHT_HANDS_MEAN'], dtype=torch.float32)
        right_hands_std = torch.tensor(stats['RIGHT_HANDS_STD'], dtype=torch.float32)
        pose_mean = torch.tensor(stats['POSE_MEAN'], dtype=torch.float32)
        pose_std = torch.tensor(stats['POSE_STD'], dtype=torch.float32)
        
        sign2ord, ord2sign, class_names = load_class_mappings('sign_language_model_mappings.pkl', format='pickle')
        
        model = SOTASignLanguageModel(
            lips_mean=lips_mean,
            lips_std=lips_std,
            left_hands_mean=left_hands_mean,
            left_hands_std=left_hands_std,
            right_hands_mean=right_hands_mean,
            right_hands_std=right_hands_std,
            pose_mean=pose_mean,
            pose_std=pose_std,
            num_classes=len(class_names)
        )
        
        weights_path = download_model_weights()
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model.eval()
            return model, ord2sign, class_names, PreprocessLayerBothHands()
        else:
            return None, None, None, None
            
    except Exception as e:
        return None, None, None, None


def load_class_mappings(load_path, format='json'):
    if format == 'pickle':
        with open(f'{load_path}', 'rb') as f:
            mappings = pickle.load(f)
    else: 
        with open(f'{load_path}', 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        mappings['ord2sign'] = {int(k): v for k, v in mappings['ord2sign'].items()}
    
    return mappings['sign2ord'], mappings['ord2sign'], mappings['class_names']


def get_data(df, preprocess_layer):
    try:
        data = load_relevant_data_subset(df)
        if data is None:
            return None
        
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        processed_data = preprocess_layer(tensor_data)
        
        if isinstance(processed_data, tuple):
            processed_data = processed_data[0]
        
        return processed_data
        
    except Exception as e:
        return None


def predict_with_text_output(model, input_tensor, ord2sign_dict, top_k=5, confidence_threshold=0.1):
    if input_tensor is None:
        return None
    
    if not hasattr(input_tensor, 'dim'):
        return None
    
    model.eval()
    
    try:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        batch_size, seq_len, num_landmarks, num_coords = input_tensor.shape
        
        non_empty_frame_idxs = []
        for b in range(batch_size):
            frame_indices = []
            for t in range(seq_len):
                frame_data = input_tensor[b, t, :, :2]
                if torch.any(frame_data != 0):
                    frame_indices.append(t)
                else:
                    frame_indices.append(-1)
            non_empty_frame_idxs.append(frame_indices)
        
        non_empty_frame_idxs = torch.tensor(non_empty_frame_idxs, dtype=torch.float32)
        
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        non_empty_frame_idxs = non_empty_frame_idxs.to(device)
        
        with torch.no_grad():
            logits = model(input_tensor, non_empty_frame_idxs, training=False)
            probabilities = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
            
            results = []
            for b in range(batch_size):
                predicted_ord = top_indices[b, 0].item()
                predicted_text = ord2sign_dict.get(predicted_ord, f"UNKNOWN_CLASS_{predicted_ord}")
                confidence = top_probs[b, 0].item()
                
                batch_result = {
                    'predicted_class_id': predicted_ord,
                    'predicted_text': predicted_text,
                    'confidence': confidence,
                    'is_confident': confidence >= confidence_threshold,
                    'top_predictions': []
                }
                
                for k in range(top_k):
                    class_id = top_indices[b, k].item()
                    class_text = ord2sign_dict.get(class_id, f"UNKNOWN_CLASS_{class_id}")
                    prob = top_probs[b, k].item()
                    
                    batch_result['top_predictions'].append({
                        'class_id': class_id,
                        'text': class_text,
                        'probability': prob
                    })
                
                results.append(batch_result)
        
        return results[0] if batch_size == 1 else results
        
    except Exception as e:
        return None


def main():
    st.set_page_config(
        page_title="Распознавание жестового языка",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 Распознавание жестового языка")
    
    model, ord2sign, class_names, preprocess_layer = load_model()
    
    if model is None:
        st.error("❌ Не удалось загрузить модель. Проверьте файлы model.py, preprocessor.py, statistics.json и _mappings.json")
        return
    
    extractor = KeypointsExtractor()
    
    mode = st.selectbox(
        "Выберите режим работы:",
        ["Загрузить видео", "Использовать веб-камеру"]
    )
    
    if mode == "Загрузить видео":
        st.subheader("📹 Загрузка видео")
        
        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm']
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                st.video(uploaded_file, width=512)
                
                if st.button("🔍 Анализировать видео"):
                    with st.spinner("Извлекаем кейпоинты из видео..."):
                        df = extractor.extract_from_video(tmp_path)
                        
                        if len(df) > 0:
                            data = get_data(df, preprocess_layer)
                            
                            if data is None:
                                return
                            
                            result = predict_with_text_output(model, data, ord2sign)
                            
                            if result is None:
                                return
                            
                            st.success("✅ Анализ завершен!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Распознанный жест", 
                                    result['predicted_text'],
                                    f"Уверенность: {result['confidence']:.2%}"
                                )
                            
                            with col2:
                                confidence_color = "🟢" if result['is_confident'] else "🟡"
                                st.metric(
                                    "Статус", 
                                    f"{confidence_color} {'Уверен' if result['is_confident'] else 'Не уверен'}",
                                    f"ID: {result['predicted_class_id']}"
                                )
                            
                            st.subheader("📊 Топ-5 предсказаний:")
                            for i, pred in enumerate(result['top_predictions'], 1):
                                st.write(f"{i}. **{pred['text']}** - {pred['probability']:.2%}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    elif mode == "Использовать веб-камеру":
        st.subheader("📷 Веб-камера")
        
        if st.button("🎥 Записать жест (3 сек)"):
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Не удалось открыть веб-камеру")
                    return
                
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", width=512)
                    
                    status_placeholder.warning(f"⏰ Запись начнется через {countdown} секунд... Приготовьтесь!")
                    time.sleep(1)
                
                frames = []
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                duration = 3
                max_frames = duration * fps
                frame_count = 0
                
                status_placeholder.info("🎬 Идет запись... Покажите жест!")
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frames.append(frame.copy())
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", width=512)
                    
                    frame_count += 1
                    
                    progress = frame_count / max_frames
                    status_placeholder.progress(progress)
                
                cap.release()
                
                if frames:
                    status_placeholder.info("🔍 Анализируем жест...")
                    
                    df = extractor.extract_from_frames(frames)
                    
                    if len(df) > 0:
                        data = get_data(df, preprocess_layer)
                        
                        if data is None:
                            return
                        
                        result = predict_with_text_output(model, data, ord2sign)
                        
                        if result is None:
                            return
                        
                        status_placeholder.success("✅ Анализ завершен!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Распознанный жест", 
                                result['predicted_text'],
                                f"Уверенность: {result['confidence']:.2%}"
                            )
                        
                        with col2:
                            confidence_color = "🟢" if result['is_confident'] else "🟡"
                            st.metric(
                                "Статус", 
                                f"{confidence_color} {'Уверен' if result['is_confident'] else 'Не уверен'}",
                                f"ID: {result['predicted_class_id']}"
                            )
                        
                        st.subheader("📊 Топ-5 предсказаний:")
                        for i, pred in enumerate(result['top_predictions'], 1):
                            st.write(f"{i}. **{pred['text']}** - {pred['probability']:.2%}")
            except Exception as e:
                st.error(f"❌ Ошибка при работе с камерой: {e}")
    
    with st.sidebar:
        st.header("ℹ️ Информация")
        st.write(f"**Загружено классов:** {len(class_names) if class_names else 'Неизвестно'}")
        st.write("**Технологии:**")
        st.write("- MediaPipe для извлечения кейпоинтов")
        st.write("- PyTorch для нейронной сети")
        st.write("- OpenCV для обработки видео")
        
        if st.button("🔄 Перезагрузить модель"):
            st.cache_resource.clear()
            st.rerun()


if __name__ == "__main__":
    main()
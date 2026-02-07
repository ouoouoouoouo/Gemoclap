"""
IEMOCAP 數據集載入器
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from config import Config



class IEMOCAPDataset(Dataset):
    """IEMOCAP 數據集類"""
    
    def __init__(self, data_list, audio_processor=None, text_processor=None, mode='train'):
        """
        Args:
            data_list: 包含 (audio_path, emotion_label, gender_label) 的列表
            audio_processor: 音頻預處理器
            text_processor: 文本預處理器
            mode: 'train' 或 'test'
        """
        self.data_list = data_list
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.mode = mode
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        if isinstance(item, dict):
            audio_path = item['audio_path']
            emotion_label = item['emotion_label']
            gender_label = item['gender_label']
        else:
            audio_path, emotion_label, gender_label = item
        
        
        # --- 載入音頻（避免 torchaudio.load → torchcodec/ffmpeg） ---
        wav, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)  # (T, C)
        # 轉單聲道
        wav = wav.mean(axis=1)  # (T,)

        # 轉 torch tensor: (1, T)
        waveform = torch.from_numpy(wav).unsqueeze(0)

        # 重採樣到 16kHz（使用 torchaudio.functional.resample，不走 torchcodec）
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # 音頻預處理
        if self.audio_processor is not None:
            # ⭐ 修改這裡:確保轉為 numpy array
            import numpy as np
            waveform_array = waveform.squeeze(0).numpy()  # 移除 batch 維度並轉為 numpy
            
            audio_input = self.audio_processor(
                waveform_array,
                sampling_rate=16000, 
                return_tensors="pt"
            )
        else:
            audio_input = {"input_values": waveform}
        
        # 獲取情緒文本描述
        emotion_text = Config.EMOTION_TEXTS[emotion_label]
        
        # 文本預處理
        if self.text_processor is not None:
            text_input = self.text_processor(
                emotion_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32
            )
        else:
            text_input = {"input_ids": torch.tensor([0])}
        
        return {
            "audio_input": audio_input,
            "text_input": text_input,
            "emotion_label": emotion_label,
            "gender_label": gender_label,
            "audio_path": audio_path
        }

def collate_fn(batch):
    """自定義批次整理函數"""
    audio_inputs = []
    text_inputs = []
    emotion_labels = []
    gender_labels = []
    audio_paths = []
    
    for item in batch:
        audio_inputs.append(item["audio_input"])
        text_inputs.append(item["text_input"])
        emotion_labels.append(item["emotion_label"])
        gender_labels.append(item["gender_label"])
        audio_paths.append(item["audio_path"])
    
    # 處理音頻輸入
    if "input_values" in audio_inputs[0]:
        max_len = max([a["input_values"].shape[-1] for a in audio_inputs])
        padded_audio = []
        attention_masks = []
        
        for audio in audio_inputs:
            input_val = audio["input_values"][0]  # 保證拿到 (T,) 或 (T) 那一維
            current_len = input_val.shape[-1]
            
            if current_len < max_len:
                padding = torch.zeros(max_len - current_len)
                padded = torch.cat([input_val, padding])
                mask = torch.cat([torch.ones(current_len), torch.zeros(max_len - current_len)])
            else:
                padded = input_val
                mask = torch.ones(current_len)
            
            padded_audio.append(padded)
            attention_masks.append(mask)
        
        audio_batch = {
            "input_values": torch.stack(padded_audio),
            "attention_mask": torch.stack(attention_masks)
        }
    else:
        audio_batch = audio_inputs
    
    # 處理文本輸入
    text_batch = {
        "input_ids": torch.cat([t["input_ids"] for t in text_inputs], dim=0),
        "attention_mask": torch.cat([t["attention_mask"] for t in text_inputs], dim=0)
    }
    
    return {
        "audio_input": audio_batch,
        "text_input": text_batch,
        "emotion_label": torch.tensor(emotion_labels, dtype=torch.long),
        "gender_label": torch.tensor(gender_labels, dtype=torch.long),
        "audio_paths": audio_paths
    }


def load_iemocap_data(iemocap_path, sessions=None):
    """
    載入 IEMOCAP 數據集
    """
    if sessions is None:
        sessions = Config.SESSIONS
    
    data_dict = {}
    
    emotion_map = {
        'neu': 'neu',
        'neutral': 'neu',
        'hap': 'hap',
        'happy': 'hap',
        'exc': 'hap',
        'excited': 'hap',
        'sad': 'sad',
        'sadness': 'sad',
        'ang': 'ang',
        'angry': 'ang',
        'anger': 'ang',
    }
    
    for session in sessions:
        session_path = os.path.join(iemocap_path, f"Session{session}")
        emo_eval_path = os.path.join(session_path, "dialog", "EmoEvaluation")
        categorical_path = os.path.join(emo_eval_path, "Categorical")
        
        if not os.path.exists(categorical_path):
            print(f"警告: {categorical_path} 不存在")
            continue
        
        for cat_file in os.listdir(categorical_path):
            if not cat_file.endswith("_cat.txt"):
                continue
            
            cat_file_path = os.path.join(categorical_path, cat_file)
            
            try:
                with open(cat_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('['):
                        continue
                    
                    parts = line.split(':')
                    if len(parts) < 2:
                        continue
                    
                    utterance_id = parts[0].strip()
                    
                    # ★★★ 添加這個過濾:只保留 improvised 對話 ★★★
                    if 'impro' not in utterance_id.lower():
                        continue
                    
                    emotions = parts[1].strip().lower()
                    
                    # 提取主要情緒
                    main_emotion = None
                    for emo_key in emotion_map.keys():
                        if emo_key in emotions:
                            main_emotion = emotion_map[emo_key]
                            break
                    
                    # 只保留需要的4個情緒類別
                    if main_emotion not in Config.EMOTION_DICT:
                        continue
                    
                    emotion_label = Config.EMOTION_DICT[main_emotion]
                    
                    # 從 utterance_id 中提取性別
                    parts_id = utterance_id.split('_')
                    if len(parts_id) >= 3:
                        gender_char = parts_id[-1][0]
                        if gender_char in Config.GENDER_DICT:
                            gender_label = Config.GENDER_DICT[gender_char]
                        else:
                            continue
                    else:
                        continue
                    
                    # 構建音頻文件路徑
                    dialog_name = '_'.join(utterance_id.split('_')[:-1])
                    audio_path = os.path.join(
                        session_path, "sentences", "wav", 
                        dialog_name, f"{utterance_id}.wav"
                    )
                    
                    if os.path.exists(audio_path):
                        # 使用 utterance_id 作為 key 來去重
                        data_dict[utterance_id] = {
                            'audio_path': audio_path,
                            'emotion_label': emotion_label,
                            'gender_label': gender_label,
                            'session': session
                        }
                    else:
                        print(f"警告: 音頻文件不存在 {audio_path}")
                        
            except Exception as e:
                print(f"處理文件 {cat_file} 時出錯: {e}")
                continue
    
    # 從字典轉換為列表
    data_list = list(data_dict.values())
    
    print(f"總共載入 {len(data_list)} 條數據 (improvised only, 去重後)")
    
    # 統計情緒分佈
    emotion_counts = {i: 0 for i in range(len(Config.EMOTION_DICT))}
    gender_counts = {i: 0 for i in range(len(Config.GENDER_DICT))}
    
    for item in data_list:
        emotion_counts[item['emotion_label']] += 1
        gender_counts[item['gender_label']] += 1
    
    print("\n情緒分佈:")
    for emo_name, emo_id in Config.EMOTION_DICT.items():
        print(f"  {emo_name}: {emotion_counts[emo_id]}")
    
    print("\n性別分佈:")
    for gen_name, gen_id in Config.GENDER_DICT.items():
        print(f"  {gen_name}: {gender_counts[gen_id]}")
    
    return data_list



def create_fold_splits(data_list, n_folds=5):
    """
    創建 5-fold 交叉驗證的分割
    按照 session 分割 (論文中的標準做法)
    
    Args:
        data_list: 數據列表 (字典格式)
        n_folds: fold 數量
    
    Returns:
        folds: [(train_data, test_data), ...] 的列表
    """
    # 將數據按 session 分組
    session_data = {i: [] for i in range(1, 6)}
    
    # 修改這裡 ↓↓↓ 處理字典格式
    for item in data_list:
        if isinstance(item, dict):
            session = item['session']
            session_data[session].append(item)
        else:
            # 向後兼容舊的元組格式
            audio_path, emotion_label, gender_label = item
            session_num = int(audio_path.split('Session')[1][0])
            session_data[session_num].append(item)
    
    # 創建 5-fold (每次留出一個 session 作為測試集)
    folds = []
    for test_session in range(1, 6):
        train_data = []
        test_data = session_data[test_session]
        
        for session_num in range(1, 6):
            if session_num != test_session:
                train_data.extend(session_data[session_num])
        
        folds.append((train_data, test_data))
        print(f"Fold {test_session}: Train={len(train_data)}, Test={len(test_data)}")
    
    return folds

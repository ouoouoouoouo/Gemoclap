"""
GEmo-CLAP 配置文件
基於論文: Gender-Attribute-Enhanced Contrastive Language-Audio Pretraining
"""

class Config:
    # 數據集設置
    IEMOCAP_PATH = "IEMOCAP_full_release"
    SESSIONS = [1, 2, 3, 4, 5]
    
    # 情緒類別映射 (論文使用4類)
    EMOTION_DICT = {
        'neu': 0,  # neutral
        'hap': 1,  # happy (包含 excited)
        'sad': 2,  # sad
        'ang': 3,  # angry
    }
    
    # 情緒文本描述 (用於文本編碼器)
    # 使用完整句子格式，更適合 CLAP 對比學習
    EMOTION_TEXTS = {
        0: "emotion is neutral",
        1: "emotion is happy", 
        2: "emotion is sad",
        3: "emotion is angry"
    }
    
    # 性別映射
    GENDER_DICT = {
        'F': 0,  # Female
        'M': 1   # Male
    }
    
    GENDER_TEXTS = {
        0: "The speaker is female.",
        1: "The speaker is male."

    }
    
    # 模型設置
    TEXT_ENCODER = "roberta-base"  # 論文使用 Roberta-base
    
    # 音頻編碼器設置
    # Large 版本（論文使用，效果最好但需要更多記憶體）
    # Base 版本（省顯存，適合記憶體不足的情況）
    AUDIO_ENCODERS = {
        # Large 版本
        "wav2vec2": "facebook/wav2vec2-large-robust",
        "hubert": "facebook/hubert-large-ll60k",
        "wavlm": "microsoft/wavlm-large",  # 論文最佳效果
        "data2vec": "facebook/data2vec-audio-large",
        
        # Base 版本 (省顯存) ↓↓↓
        "wav2vec2-base": "facebook/wav2vec2-base",
        "hubert-base": "facebook/hubert-base-ls960",
        "wavlm-base": "microsoft/wavlm-base",
        "wavlm-base-plus": "microsoft/wavlm-base-plus",
    }
    
    # 預設使用 WavLM-large (論文中效果最好)
    # 如果記憶體不足，可以改用 base 版本：
    # - "wavlm-base" 或 "wavlm-base-plus" (推薦)
    # - "wav2vec2-base" (最小，但效果稍差)
    
    
    # 訓練超參數
    # ⚠️ 對比學習需要足夠大的 batch size！
    BATCH_SIZE = 8 # 根據顯存調整
    GRADIENT_ACCUMULATION_STEPS = 4 # 有效 batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    LEARNING_RATE = 2e-5  # 論文標準學習率
    NUM_EPOCHS = 80
    EMBEDDING_DIM = 512  # 投影層輸出維度
    
    
    INIT_TAU = 0.07
    LOGIT_SCALE_MIN = 1e-3
    LOGIT_SCALE_MAX = 100.0

    
    # 多任務學習權重 (ML-GEmo-CLAP)
    ALPHA_E = 0.9  # 情緒損失權重 (0.8 或 0.9)
    
    # 軟標籤權重 (SL-GEmo-CLAP)
    ALPHA_E_SL = 0.8  # 情緒矩陣權重
    
    # 其他設置
    NUM_WORKERS = 0  # Windows 上建議設為 0
    SEED = 42
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # 評估指標
    METRICS = ["WAR", "UAR"]  # Weighted Average Recall, Unweighted Average Recall
    
    # 5-fold 交叉驗證
    N_FOLDS = 5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaModel, 
    Wav2Vec2Model, 
    HubertModel,
    WavLMModel,
    Data2VecAudioModel
)
from config import Config

class ProjectionHead(nn.Module):
    """MLP 投影頭"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.proj(x)


class EmoCLAP(nn.Module):
    """基線 Emo-CLAP 模型（使用可學習溫度參數）"""
    
    def __init__(self, text_encoder_name, audio_encoder_name, embedding_dim=512):
        super().__init__()
        
        # 文本編碼器
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        text_hidden_dim = self.text_encoder.config.hidden_size
        
        # 音頻編碼器
        audio_encoder_map = {
            "wav2vec2": Wav2Vec2Model,
            "wav2vec2-base": Wav2Vec2Model, 
            "hubert": HubertModel,
            "hubert-base": HubertModel,  
            "wavlm": WavLMModel,
            "wavlm-base": WavLMModel, 
            "wavlm-base-plus": WavLMModel,  
            "data2vec": Data2VecAudioModel
        }
        
        encoder_class = audio_encoder_map.get(audio_encoder_name)
        if encoder_class is None:
            raise ValueError(f"不支持的音頻編碼器: {audio_encoder_name}")
        
        encoder_path = Config.AUDIO_ENCODERS[audio_encoder_name]
        self.audio_encoder = encoder_class.from_pretrained(encoder_path)
        audio_hidden_dim = self.audio_encoder.config.hidden_size
        
        # 投影頭
        self.text_projection = ProjectionHead(text_hidden_dim, embedding_dim)
        self.audio_projection = ProjectionHead(audio_hidden_dim, embedding_dim)
        
        # ⚠️ 可學習的溫度參數（log space）
        # 初始化為 log(1/0.07) ≈ 2.66
        self.logit_scale_audio = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
        self.embedding_dim = embedding_dim
    
    def encode_text(self, text_input):
        """編碼文本"""
        outputs = self.text_encoder(**text_input)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        return text_embeddings
    
    def encode_audio(self, audio_input):
        """編碼音頻（使用近似的 masked mean pooling）"""
        outputs = self.audio_encoder(**audio_input)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len_downsampled, hidden_dim)
        
        # ⭐ 修復：處理下採樣後的 attention mask
        if "attention_mask" in audio_input and audio_input["attention_mask"] is not None:
            # 獲取原始和下採樣後的長度
            original_length = audio_input["attention_mask"].shape[1]  # 原始音頻長度
            downsampled_length = hidden_states.shape[1]  # 下採樣後的長度
            
            # 計算下採樣比率
            downsample_ratio = original_length / downsampled_length
            
            # 計算每個樣本的有效長度（下採樣後）
            input_lengths = audio_input["attention_mask"].sum(dim=1).float()  # (batch_size,)
            output_lengths = (input_lengths / downsample_ratio).long()  # (batch_size,)
            
            # 創建下採樣後的 mask
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # 生成序列索引: (1, seq_len)
            seq_range = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
            # 比較生成 mask: (batch_size, seq_len)
            mask = (seq_range < output_lengths.unsqueeze(1)).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            
            # Masked mean pooling
            sum_hidden = (hidden_states * mask).sum(dim=1)  # (batch_size, hidden_dim)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # (batch_size, 1)
            audio_features = sum_hidden / sum_mask  # (batch_size, hidden_dim)
        else:
            # 沒有 attention_mask，使用簡單 mean pooling
            audio_features = hidden_states.mean(dim=1)
        
        audio_embeddings = self.audio_projection(audio_features)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
        return audio_embeddings
    
    def forward(self, audio_input, text_input):
        """前向傳播"""
        audio_embeddings = self.encode_audio(audio_input)
        text_embeddings = self.encode_text(text_input)
        return audio_embeddings, text_embeddings
    
    def compute_contrastive_loss(self, audio_embeddings, text_embeddings, emotion_labels):
        """
        計算對比損失 (論文公式 4)
        """
        batch_size = audio_embeddings.shape[0]
        
        # 構建情緒 ground truth 矩陣
        emotion_labels_expanded = emotion_labels.unsqueeze(1)
        M_e = (emotion_labels_expanded == emotion_labels_expanded.T).float()
        
        # ⚠️ 在 log space 限制範圍（更穩定）
        log_scale_a = torch.clamp(self.logit_scale_audio, min=np.log(1.0), max=np.log(100.0))
        log_scale_t = torch.clamp(self.logit_scale_text, min=np.log(1.0), max=np.log(100.0))
        
        # 轉換到實際的 scale
        scale_audio = log_scale_a.exp()
        scale_text = log_scale_t.exp()
        
        # 計算相似度矩陣（乘法形式）
        C_audio = scale_audio * torch.matmul(audio_embeddings, text_embeddings.T)
        C_text = scale_text * torch.matmul(text_embeddings, audio_embeddings.T)
        
        # log_softmax
        log_S_audio = F.log_softmax(C_audio, dim=-1)
        log_S_text = F.log_softmax(C_text, dim=-1)
        
        # Ground truth 做 softmax
        S_M_e = F.softmax(M_e, dim=-1)
        
        # KL 散度
        kl_audio = F.kl_div(log_S_audio, S_M_e, reduction='batchmean')
        kl_text = F.kl_div(log_S_text, S_M_e, reduction='batchmean')
        
        # 總損失
        loss = 0.5 * (kl_audio + kl_text)
        
        return loss


class MLGEmoCLAP(EmoCLAP):
    """ML-GEmo-CLAP: 多任務學習版本"""
    
    def __init__(self, text_encoder_name, audio_encoder_name, embedding_dim=512, alpha_e=0.9):
        super().__init__(text_encoder_name, audio_encoder_name, embedding_dim)
        self.alpha_e = alpha_e
    
    def compute_multitask_loss(self, audio_embeddings, text_embeddings, 
                               emotion_labels, gender_labels):
        """計算多任務損失 (論文公式 6)"""
        batch_size = audio_embeddings.shape[0]
        
        # 構建矩陣
        emotion_labels_expanded = emotion_labels.unsqueeze(1)
        M_emotion = (emotion_labels_expanded == emotion_labels_expanded.T).float()
        
        gender_labels_expanded = gender_labels.unsqueeze(1)
        M_gender = (gender_labels_expanded == gender_labels_expanded.T).float()
        
        # ⚠️ 在 log space 限制範圍
        log_scale_a = torch.clamp(self.logit_scale_audio, min=np.log(1.0), max=np.log(100.0))
        log_scale_t = torch.clamp(self.logit_scale_text, min=np.log(1.0), max=np.log(100.0))
        
        scale_audio = log_scale_a.exp()
        scale_text = log_scale_t.exp()
        
        # 計算相似度矩陣
        C_audio = scale_audio * torch.matmul(audio_embeddings, text_embeddings.T)
        C_text = scale_text * torch.matmul(text_embeddings, audio_embeddings.T)
        
        # log_softmax
        log_S_audio = F.log_softmax(C_audio, dim=-1)
        log_S_text = F.log_softmax(C_text, dim=-1)
        
        # Ground truth 做 softmax
        S_emotion = F.softmax(M_emotion, dim=-1)
        S_gender = F.softmax(M_gender, dim=-1)
        
        # 情緒損失
        kl_emotion_audio = F.kl_div(log_S_audio, S_emotion, reduction='batchmean')
        kl_emotion_text = F.kl_div(log_S_text, S_emotion, reduction='batchmean')
        L_emotion = 0.5 * (kl_emotion_audio + kl_emotion_text)
        
        # 性別損失
        kl_gender_audio = F.kl_div(log_S_audio, S_gender, reduction='batchmean')
        kl_gender_text = F.kl_div(log_S_text, S_gender, reduction='batchmean')
        L_gender = 0.5 * (kl_gender_audio + kl_gender_text)
        
        # 組合損失
        loss = self.alpha_e * L_emotion + (1 - self.alpha_e) * L_gender
        
        return loss


class SLGEmoCLAP(EmoCLAP):
    """SL-GEmo-CLAP: 軟標籤版本"""
    
    def __init__(self, text_encoder_name, audio_encoder_name, embedding_dim=512, alpha_e=0.9):
        super().__init__(text_encoder_name, audio_encoder_name, embedding_dim)
        self.alpha_e = alpha_e
    
    def compute_soft_label_loss(self, audio_embeddings, text_embeddings,
                                emotion_labels, gender_labels):
        """計算軟標籤損失 (論文公式 7)"""
        batch_size = audio_embeddings.shape[0]
        
        # 構建矩陣
        emotion_labels_expanded = emotion_labels.unsqueeze(1)
        M_emotion = (emotion_labels_expanded == emotion_labels_expanded.T).float()
        
        gender_labels_expanded = gender_labels.unsqueeze(1)
        M_gender = (gender_labels_expanded == gender_labels_expanded.T).float()
        
        # 組合矩陣
        M_combined = self.alpha_e * M_emotion + (1 - self.alpha_e) * M_gender
        
        # ⚠️ 在 log space 限制範圍
        log_scale_a = torch.clamp(self.logit_scale_audio, min=np.log(1.0), max=np.log(100.0))
        log_scale_t = torch.clamp(self.logit_scale_text, min=np.log(1.0), max=np.log(100.0))
        
        scale_audio = log_scale_a.exp()
        scale_text = log_scale_t.exp()
        
        # 計算相似度矩陣
        C_audio = scale_audio * torch.matmul(audio_embeddings, text_embeddings.T)
        C_text = scale_text * torch.matmul(text_embeddings, audio_embeddings.T)
        
        # log_softmax
        log_S_audio = F.log_softmax(C_audio, dim=-1)
        log_S_text = F.log_softmax(C_text, dim=-1)
        
        # Ground truth 做 softmax
        S_combined = F.softmax(M_combined, dim=-1)
        
        # KL 散度
        kl_audio = F.kl_div(log_S_audio, S_combined, reduction='batchmean')
        kl_text = F.kl_div(log_S_text, S_combined, reduction='batchmean')
        
        # 總損失
        loss = 0.5 * (kl_audio + kl_text)
        
        return loss


def create_model(model_type="emo_clap", text_encoder=None, audio_encoder=None, 
                 embedding_dim=512, alpha_e=0.9):
    """創建模型的工廠函數"""
    if text_encoder is None:
        text_encoder = Config.TEXT_ENCODER
    if audio_encoder is None:
        audio_encoder = Config.AUDIO_ENCODER
    
    if model_type.lower() == "emo_clap":
        model = EmoCLAP(text_encoder, audio_encoder, embedding_dim)
    elif model_type.lower() == "ml_gemo_clap":
        model = MLGEmoCLAP(text_encoder, audio_encoder, embedding_dim, alpha_e)
    elif model_type.lower() == "sl_gemo_clap":
        model = SLGEmoCLAP(text_encoder, audio_encoder, embedding_dim, alpha_e)
    else:
        raise ValueError(f"未知的模型類型: {model_type}")
    
    return model
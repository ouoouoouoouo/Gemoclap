"""
GEmo-CLAP 推論腳本
用於對單個音頻文件進行情緒識別
"""

import torch
import torchaudio
import argparse
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from config import Config
from models import create_model


def load_audio(audio_path):
    """載入並預處理音頻"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 轉為單聲道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重採樣到 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy()


def predict_emotion(model, audio_path, audio_processor, text_processor, device):
    """預測音頻的情緒"""
    model.eval()
    
    # 載入音頻
    audio_array = load_audio(audio_path)
    
    # 預處理音頻
    audio_input = audio_processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    audio_input = {k: v.to(device) for k, v in audio_input.items()}
    
    # 為每個情緒類別創建文本嵌入
    with torch.no_grad():
        emotion_text_embeddings = []
        
        for emotion_id in range(len(Config.EMOTION_DICT)):
            emotion_text = Config.EMOTION_TEXTS[emotion_id]
            text_input = text_processor(
                emotion_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32
            )
            text_input = {k: v.to(device) for k, v in text_input.items()}
            
            # 編碼文本
            text_emb = model.encode_text(text_input)
            emotion_text_embeddings.append(text_emb)
        
        # 堆疊成 [num_classes, embedding_dim]
        emotion_text_embeddings = torch.cat(emotion_text_embeddings, dim=0)
        
        # 編碼音頻
        audio_embedding = model.encode_audio(audio_input)
        
        # 計算與每個情緒文本的相似度
        similarities = torch.matmul(audio_embedding, emotion_text_embeddings.T)
        
        # 獲取預測結果和置信度
        probabilities = torch.softmax(similarities, dim=-1)
        prediction = torch.argmax(similarities, dim=-1)
        confidence = probabilities[0, prediction].item()
    
    # 獲取情緒名稱
    emotion_names = {v: k for k, v in Config.EMOTION_DICT.items()}
    predicted_emotion = emotion_names[prediction.item()]
    
    # 所有情緒的置信度
    emotion_scores = {}
    for i, emotion_name in emotion_names.items():
        emotion_scores[emotion_name] = probabilities[0, i].item()
    
    return predicted_emotion, confidence, emotion_scores


def main(args):
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU 加速: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  使用 CPU（較慢）")
    
    # 載入處理器
    print("\n載入預訓練處理器...")
    text_processor = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
    audio_encoder_path = Config.AUDIO_ENCODERS[args.audio_encoder]
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
    
    # 創建模型
    print("\n創建模型...")
    model = create_model(
        model_type=args.model_type,
        text_encoder=Config.TEXT_ENCODER,
        audio_encoder=args.audio_encoder,
        embedding_dim=Config.EMBEDDING_DIM,
        alpha_e=Config.ALPHA_E if args.model_type != "emo_clap" else None
    )
    
    # 載入模型權重
    print(f"\n載入模型權重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # 預測
    print(f"\n分析音頻: {args.audio_path}")
    predicted_emotion, confidence, emotion_scores = predict_emotion(
        model, args.audio_path, audio_processor, text_processor, device
    )
    
    # 打印結果
    print(f"\n{'='*50}")
    print(f"預測結果")
    print(f"{'='*50}")
    print(f"預測情緒: {predicted_emotion}")
    print(f"置信度: {confidence*100:.2f}%")
    
    print(f"\n所有情緒的置信度:")
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_emotions:
        bar = "█" * int(score * 50)
        print(f"  {emotion:>6}: {score*100:>5.2f}% {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 GEmo-CLAP 進行情緒識別")
    
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="音頻文件路徑"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型檢查點路徑"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sl_gemo_clap",
        choices=["emo_clap", "ml_gemo_clap", "sl_gemo_clap"],
        help="模型類型"
    )
    parser.add_argument(
        "--audio_encoder",
        type=str,
        default="wavlm",
        choices=["wav2vec2", "hubert", "wavlm", "data2vec"],
        help="音頻編碼器類型"
    )
    
    args = parser.parse_args()
    main(args)


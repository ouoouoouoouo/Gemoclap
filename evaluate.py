"""
GEmo-CLAP 評估腳本
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from tqdm import tqdm

from config import Config


def evaluate_model(model, dataloader, device):
    """
    評估模型性能
    """
    model.eval()
    # ⭐ 添加這行：獲取實際的模型（處理 DataParallel）
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    all_predictions = []
    all_labels = []
    
    # 為每個情緒類別創建文本嵌入
    with torch.no_grad():
        emotion_text_embeddings = []
        
        from transformers import AutoTokenizer
        text_processor = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
        
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
            text_emb = actual_model.encode_text(text_input)
            emotion_text_embeddings.append(text_emb)
        
        # 堆疊成 [num_classes, embedding_dim]
        emotion_text_embeddings = torch.cat(emotion_text_embeddings, dim=0)
    
    # 評估
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="評估中")
        for batch in pbar:
            audio_input = {k: v.to(device) for k, v in batch["audio_input"].items()}
            emotion_labels = batch["emotion_label"].to(device)
            
            # 編碼音頻
            audio_embeddings = actual_model.encode_audio(audio_input)
            
            # 計算與每個情緒文本的相似度
            similarities = torch.matmul(audio_embeddings, emotion_text_embeddings.T)
            
            # 獲取預測結果
            predictions = torch.argmax(similarities, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(emotion_labels.cpu().numpy())
    
    # 轉換為 numpy 數組
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # ★★★ 添加診斷信息 ★★★
    from collections import Counter
    pred_dist = Counter(all_predictions)
    label_dist = Counter(all_labels)
    
    print("\n" + "="*50)
    print("診斷信息")
    print("="*50)
    
    print("\n預測分佈:")
    emotion_names = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}
    for emo_id in range(4):
        emo_name = emotion_names[emo_id]
        pred_count = pred_dist.get(emo_id, 0)
        pred_pct = (pred_count / len(all_predictions)) * 100 if len(all_predictions) > 0 else 0
        print(f"  {emo_name}: {pred_count:4d} ({pred_pct:5.1f}%)")
    
    print("\n真實標籤分佈:")
    for emo_id in range(4):
        emo_name = emotion_names[emo_id]
        label_count = label_dist.get(emo_id, 0)
        label_pct = (label_count / len(all_labels)) * 100 if len(all_labels) > 0 else 0
        print(f"  {emo_name}: {label_count:4d} ({label_pct:5.1f}%)")
    
    # 檢查是否只預測單一類別
    unique_predictions = len(pred_dist)
    if unique_predictions == 1:
        print(f"\n⚠️  警告: 模型只預測了 1 個類別!")
        print(f"   這通常意味著:")
        print(f"   1. 學習率太小,模型沒有真正學習")
        print(f"   2. 損失函數有問題")
        print(f"   3. 特徵沒有正確提取")
    elif unique_predictions == 2:
        print(f"\n⚠️  警告: 模型只預測了 2 個類別,可能訓練不足")
    
    print("="*50 + "\n")
    
    # 計算 WAR (Weighted Average Recall / Accuracy)
    war = accuracy_score(all_labels, all_predictions) * 100
    
    # 計算 UAR (Unweighted Average Recall)
    uar = recall_score(all_labels, all_predictions, average='macro') * 100
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 計算每個類別的召回率
    class_recalls = []
    for i in range(len(Config.EMOTION_DICT)):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
            class_recalls.append(recall)
        else:
            class_recalls.append(0.0)
    
    metrics = {
        "WAR": war,
        "UAR": uar,
        "confusion_matrix": cm,
        "class_recalls": class_recalls,
        "predictions": all_predictions,
        "labels": all_labels
    }
    
    return metrics

def print_evaluation_results(metrics):
    """打印評估結果"""
    print(f"\n{'='*50}")
    print(f"評估結果")
    print(f"{'='*50}")
    print(f"WAR (Weighted Average Recall): {metrics['WAR']:.2f}%")
    print(f"UAR (Unweighted Average Recall): {metrics['UAR']:.2f}%")
    
    print(f"\n每個類別的召回率:")
    emotion_names = {v: k for k, v in Config.EMOTION_DICT.items()}
    for i, recall in enumerate(metrics['class_recalls']):
        emotion_name = emotion_names.get(i, f"Class_{i}")
        print(f"  {emotion_name}: {recall*100:.2f}%")
    
    print(f"\n混淆矩陣:")
    cm = metrics['confusion_matrix']
    print("     ", end="")
    for i in range(len(Config.EMOTION_DICT)):
        emotion_name = emotion_names.get(i, f"C{i}")
        print(f"{emotion_name:>6}", end="")
    print()
    
    for i in range(len(Config.EMOTION_DICT)):
        emotion_name = emotion_names.get(i, f"C{i}")
        print(f"{emotion_name:>6}", end="")
        for j in range(len(Config.EMOTION_DICT)):
            print(f"{cm[i,j]:>6}", end="")
        print()


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor
    
    from dataset import IEMOCAPDataset, load_iemocap_data, create_fold_splits, collate_fn
    from models import create_model
    
    parser = argparse.ArgumentParser(description="評估 GEmo-CLAP 模型")
    
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
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="評估哪個 fold"
    )
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU 加速: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  使用 CPU（較慢）")
    
    # 載入數據
    print("\n載入 IEMOCAP 數據集...")
    data_list = load_iemocap_data(Config.IEMOCAP_PATH)
    
    # 創建 fold 分割
    print("\n創建 5-fold 交叉驗證分割...")
    folds = create_fold_splits(data_list, n_folds=Config.N_FOLDS)
    
    # 獲取測試集
    _, test_data = folds[args.fold - 1]
    
    # 初始化處理器
    print("\n載入預訓練處理器...")
    audio_encoder_path = Config.AUDIO_ENCODERS[args.audio_encoder]
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
    
    from transformers import AutoTokenizer
    text_processor = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
    
    # 創建數據集
    test_dataset = IEMOCAPDataset(
        test_data,
        audio_processor=audio_processor,
        text_processor=text_processor,
        mode='test'
    )
    
    # 創建數據加載器
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
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
    
    # 評估
    metrics = evaluate_model(model, test_loader, device)
    
    # 打印結果
    print_evaluation_results(metrics)


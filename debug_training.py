"""
調試訓練數據載入問題
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from collections import Counter

from config import Config
from dataset import (
    IEMOCAPDataset, 
    load_iemocap_data, 
    create_fold_splits,
    collate_fn
)

print("="*60)
print("調試訓練數據載入")
print("="*60)

# 載入數據
print("\n1. 載入 IEMOCAP 數據...")
data = load_iemocap_data(Config.IEMOCAP_PATH)

print(f"\n總樣本數: {len(data)}")

# ⚠️ 檢查數據格式
print("\n檢查數據格式:")
if len(data) > 0:
    first_item = data[0]
    print(f"  第一個樣本的類型: {type(first_item)}")
    if isinstance(first_item, dict):
        print(f"  字典的 keys: {first_item.keys()}")
    elif isinstance(first_item, (tuple, list)):
        print(f"  Tuple/List 長度: {len(first_item)}")

# ⚠️ 根據數據格式提取標籤（使用正確的 key）
print("\n情緒分佈:")
if isinstance(data[0], dict):
    # 如果是字典格式，使用 'emotion_label' 而不是 'emotion'
    emotion_dist = Counter([d['emotion_label'] for d in data])
    gender_dist = Counter([d['gender_label'] for d in data])
else:
    # 如果是 tuple 格式
    emotion_dist = Counter([d[1] for d in data])
    gender_dist = Counter([d[2] for d in data])

for emotion, count in sorted(emotion_dist.items()):
    emotion_name = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}[emotion]
    print(f"  {emotion_name}: {count}")

print("\n性別分佈:")
for gender, count in sorted(gender_dist.items()):
    gender_name = {0: 'F', 1: 'M'}[gender]
    print(f"  {gender_name}: {count}")

# 創建 5-fold splits
print("\n2. 創建 5-fold 交叉驗證分割...")
fold_splits = create_fold_splits(data, n_folds=Config.N_FOLDS)

# 檢查每個 fold 的數據量
print("\n3. 檢查每個 fold:")
for fold_num, (train_data, test_data) in enumerate(fold_splits, 1):
    print(f"  Fold {fold_num}: Train={len(train_data)}, Test={len(test_data)}")

# ⭐⭐⭐ 詳細檢查每個 Fold 的類別分布 ⭐⭐⭐
print("\n" + "="*60)
print("檢查每個 Fold 的類別分布")
print("="*60)

for fold_num, (train_data, test_data) in enumerate(fold_splits, 1):
    print(f"\nFold {fold_num} (Session {fold_num} 作為測試集):")
    
    # ⚠️ 使用正確的 key
    if isinstance(test_data[0], dict):
        test_emotions = Counter([d['emotion_label'] for d in test_data])
        train_emotions = Counter([d['emotion_label'] for d in train_data])
    else:
        test_emotions = Counter([d[1] for d in test_data])
        train_emotions = Counter([d[1] for d in train_data])
    
    # 測試集分布
    print(f"  測試集大小: {len(test_data)}")
    print(f"  測試集類別分布:")
    for emotion_id in sorted(test_emotions.keys()):
        emotion_name = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}[emotion_id]
        count = test_emotions[emotion_id]
        percentage = count / len(test_data) * 100
        print(f"    {emotion_name}: {count:4d} ({percentage:5.1f}%)")
    
    # 計算測試集的類別不平衡程度
    if len(test_emotions) > 0:
        max_count = max(test_emotions.values())
        min_count = min(test_emotions.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  測試集不平衡比例: {imbalance_ratio:.2f}:1 (最大類/最小類)")
        
        if imbalance_ratio > 3.0:
            print(f"  ⚠️  警告: 測試集類別嚴重不平衡 (>{3.0}:1)")
    
    # 訓練集分布
    print(f"\n  訓練集大小: {len(train_data)}")
    print(f"  訓練集類別分布:")
    for emotion_id in sorted(train_emotions.keys()):
        emotion_name = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}[emotion_id]
        count = train_emotions[emotion_id]
        percentage = count / len(train_data) * 100
        print(f"    {emotion_name}: {count:4d} ({percentage:5.1f}%)")
    
    # 計算訓練集的類別不平衡程度
    if len(train_emotions) > 0:
        max_count = max(train_emotions.values())
        min_count = min(train_emotions.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  訓練集不平衡比例: {imbalance_ratio:.2f}:1 (最大類/最小類)")
        
        if imbalance_ratio > 3.0:
            print(f"  ⚠️  警告: 訓練集類別嚴重不平衡 (>{3.0}:1)")

# 重點檢查 Fold 1（你的最好 fold）
print("\n" + "="*60)
print("詳細檢查 Fold 1 (表現最好 - 75.74%)")
print("="*60)
fold_num = 1
train_data, test_data = fold_splits[fold_num - 1]

print(f"訓練集樣本數: {len(train_data)}")
print(f"驗證集樣本數: {len(test_data)}")

# 創建 Dataset
print("\n5. 創建訓練 Dataset...")
text_processor = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
    Config.AUDIO_ENCODERS["wavlm"]
)

train_dataset = IEMOCAPDataset(
    train_data,
    audio_processor=audio_processor,
    text_processor=text_processor,
    mode='train'
)

print(f"  Dataset 長度: {len(train_dataset)}")

# 創建 DataLoader
print("\n6. 創建訓練 DataLoader...")
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # 調試時設為 0
    collate_fn=collate_fn,
    pin_memory=True
)

print(f"  DataLoader 長度 (batch 數量): {len(train_loader)}")
print(f"  Batch Size: {Config.BATCH_SIZE}")
print(f"  預期樣本數: {len(train_loader)} × {Config.BATCH_SIZE} ≈ {len(train_loader) * Config.BATCH_SIZE}")

# 檢查第一個 batch
print("\n7. 檢查第一個 batch:")
try:
    first_batch = next(iter(train_loader))
    print(f"  audio_input shape: {first_batch['audio_input']['input_values'].shape}")
    print(f"  text_input shape: {first_batch['text_input']['input_ids'].shape}")
    print(f"  emotion_label shape: {first_batch['emotion_label'].shape}")
    print(f"  gender_label shape: {first_batch['gender_label'].shape}")
    print(f"  實際 batch size: {len(first_batch['emotion_label'])}")
    
    # 檢查第一個 batch 的情緒分布
    batch_emotions = Counter(first_batch['emotion_label'].tolist())
    print(f"\n  第一個 batch 的情緒分布:")
    for emotion_id in sorted(batch_emotions.keys()):
        emotion_name = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}[emotion_id]
        count = batch_emotions[emotion_id]
        print(f"    {emotion_name}: {count}")
except Exception as e:
    print(f"  ⚠️  錯誤: {e}")
    import traceback
    traceback.print_exc()

# 計算有效 batch size
print("\n8. 梯度累積設置:")
print(f"  BATCH_SIZE: {Config.BATCH_SIZE}")
print(f"  GRADIENT_ACCUMULATION_STEPS: {Config.GRADIENT_ACCUMULATION_STEPS}")
print(f"  有效 batch size: {Config.BATCH_SIZE} × {Config.GRADIENT_ACCUMULATION_STEPS} = {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
print(f"  梯度更新次數: {len(train_loader)} ÷ {Config.GRADIENT_ACCUMULATION_STEPS} = {len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS}")

# ⭐ Fold 難度分析
print("\n" + "="*60)
print("Fold 難度分析（基於類別不平衡）")
print("="*60)

fold_difficulties = []
for fold_num, (train_data, test_data) in enumerate(fold_splits, 1):
    # ⚠️ 使用正確的 key
    if isinstance(test_data[0], dict):
        test_emotions = Counter([d['emotion_label'] for d in test_data])
        train_emotions = Counter([d['emotion_label'] for d in train_data])
    else:
        test_emotions = Counter([d[1] for d in test_data])
        train_emotions = Counter([d[1] for d in train_data])
    
    # 計算不平衡比例
    if len(test_emotions) > 0 and len(train_emotions) > 0:
        test_imbalance = max(test_emotions.values()) / min(test_emotions.values())
        train_imbalance = max(train_emotions.values()) / min(train_emotions.values())
        difficulty_score = (test_imbalance + train_imbalance) / 2
        
        fold_difficulties.append({
            'fold': fold_num,
            'test_imbalance': test_imbalance,
            'train_imbalance': train_imbalance,
            'difficulty': difficulty_score,
            'test_size': len(test_data),
            'train_size': len(train_data)
        })

# 按難度排序
fold_difficulties.sort(key=lambda x: x['difficulty'], reverse=True)

print("\nFold 難度排名（從難到易）:")
for i, fold_info in enumerate(fold_difficulties, 1):
    fold_num = fold_info['fold']
    difficulty = fold_info['difficulty']
    test_imb = fold_info['test_imbalance']
    train_imb = fold_info['train_imbalance']
    test_size = fold_info['test_size']
    train_size = fold_info['train_size']
    
    print(f"\n{i}. Fold {fold_num}:")
    print(f"   訓練集: {train_size} 樣本, 不平衡: {train_imb:.2f}:1")
    print(f"   測試集: {test_size} 樣本, 不平衡: {test_imb:.2f}:1")
    print(f"   綜合難度分數: {difficulty:.2f}")
    
    # 標記預期表現
    if difficulty > 3.0:
        print(f"   ⚠️  預期表現: 差 (UAR 可能 < 70%)")
    elif difficulty > 2.5:
        print(f"   ⚡ 預期表現: 中等 (UAR 可能 70-75%)")
    else:
        print(f"   ✅ 預期表現: 好 (UAR 可能 > 75%)")

# 對比實際結果
print("\n" + "="*60)
print("對比實際訓練結果")
print("="*60)

actual_results = {
    1: 75.74,
    2: 73.95,
    3: 73.93,
    4: 65.99,
    5: 67.61
}

print("\n難度預測 vs 實際表現:")
for fold_info in fold_difficulties:
    fold_num = fold_info['fold']
    predicted_difficulty = fold_info['difficulty']
    actual_uar = actual_results.get(fold_num, 0)
    
    print(f"\nFold {fold_num}:")
    print(f"  預測難度: {predicted_difficulty:.2f} (難度排名 #{fold_difficulties.index(fold_info) + 1})")
    print(f"  實際 UAR: {actual_uar:.2f}%")
    
    # 判斷預測是否準確
    if predicted_difficulty > 3.0 and actual_uar < 70:
        print(f"  ✅ 預測準確：高難度 fold 表現差")
    elif predicted_difficulty < 2.5 and actual_uar > 75:
        print(f"  ✅ 預測準確：低難度 fold 表現好")
    elif 2.5 <= predicted_difficulty <= 3.0 and 70 <= actual_uar <= 75:
        print(f"  ✅ 預測準確：中等難度 fold 表現中等")
    else:
        print(f"  ⚠️  預測與實際有偏差")

print("\n" + "="*60)
print("調試完成")
print("="*60)
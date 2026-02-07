# check_all_folds.py
from config import Config
from dataset import load_iemocap_data, create_fold_splits
from collections import Counter

data = load_iemocap_data(Config.IEMOCAP_PATH)
folds = create_fold_splits(data, n_folds=5)

print("="*70)
print("所有 Fold 的 angry 類別樣本數")
print("="*70)

emotion_names = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}

print(f"\n{'Fold':<6} {'訓練集 ang':<12} {'測試集 ang':<12} {'測試集佔比':<12} {'實際UAR'}")
print("-" * 70)

actual_results = {1: 75.74, 2: 73.95, 3: 73.93, 4: 65.99, 5: 67.61}

for fold_num in range(1, 6):
    train_data, test_data = folds[fold_num - 1]
    
    train_emotions = Counter([item['emotion_label'] for item in train_data])
    test_emotions = Counter([item['emotion_label'] for item in test_data])
    
    # angry 的 ID 是 3
    train_ang = train_emotions.get(3, 0)
    test_ang = test_emotions.get(3, 0)
    test_ang_pct = (test_ang / len(test_data)) * 100
    
    actual_uar = actual_results[fold_num]
    
    print(f"{fold_num:<6} {train_ang:<12} {test_ang:<12} {test_ang_pct:<12.1f}% {actual_uar:.2f}%")

print("\n" + "="*70)
print("每個 Fold 的完整測試集分布")
print("="*70)

for fold_num in range(1, 6):
    train_data, test_data = folds[fold_num - 1]
    test_emotions = Counter([item['emotion_label'] for item in test_data])
    
    print(f"\nFold {fold_num} 測試集:")
    min_count = min(test_emotions.values())
    
    for emotion_id in sorted(test_emotions.keys()):
        emotion_name = emotion_names[emotion_id]
        count = test_emotions[emotion_id]
        percentage = (count / len(test_data)) * 100
        
        marker = ""
        if count == min_count:
            marker = " ← 最小類"
        
        print(f"  {emotion_name}: {count:4d} ({percentage:5.1f}%){marker}")
    
    actual_uar = actual_results[fold_num]
    print(f"  實際 UAR: {actual_uar:.2f}%")

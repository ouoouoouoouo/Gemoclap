"""
檢查 Fold 2 的詳細情況
"""
from config import Config
from dataset import load_iemocap_data, create_fold_splits
from collections import Counter

data = load_iemocap_data(Config.IEMOCAP_PATH)
folds = create_fold_splits(data, n_folds=5)

print("="*60)
print("所有 Fold 的詳細類別分布")
print("="*60)

for fold_num in range(1, 6):
    train_data, test_data = folds[fold_num - 1]
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_num} (Session {fold_num} 作為測試集)")
    print(f"{'='*60}")
    
    # 測試集分布
    test_emotions = Counter([item['emotion_label'] for item in test_data])
    print(f"\n測試集 ({len(test_data)} 樣本):")
    
    emotion_names = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}
    test_details = {}
    
    for emotion_id in sorted(test_emotions.keys()):
        emotion_name = emotion_names[emotion_id]
        count = test_emotions[emotion_id]
        percentage = count / len(test_data) * 100
        test_details[emotion_name] = count
        print(f"  {emotion_name}: {count:4d} ({percentage:5.1f}%)")
    
    # 計算不平衡比例
    max_count = max(test_emotions.values())
    min_count = min(test_emotions.values())
    imbalance = max_count / min_count
    
    print(f"\n  最大類: {max_count} 樣本")
    print(f"  最小類: {min_count} 樣本")
    print(f"  不平衡比例: {imbalance:.2f}:1")
    
    # 訓練集分布
    train_emotions = Counter([item['emotion_label'] for item in train_data])
    print(f"\n訓練集 ({len(train_data)} 樣本):")
    
    for emotion_id in sorted(train_emotions.keys()):
        emotion_name = emotion_names[emotion_id]
        count = train_emotions[emotion_id]
        percentage = count / len(train_data) * 100
        print(f"  {emotion_name}: {count:4d} ({percentage:5.1f}%)")
    
    # 理論 UAR 分析
    print(f"\n理論 UAR 分析:")
    print(f"  如果模型完全隨機: UAR = 25.00%")
    
    # 如果只預測最大類
    max_class_uar = 100.0 / len(test_emotions)  # 只有最大類是 100%，其他都是 0%
    print(f"  如果只預測最大類: UAR = {max_class_uar:.2f}%")
    
    # 如果按比例隨機
    # 每個類的期望 recall = 該類在測試集中的比例
    proportional_uar = 100.0 / len(test_emotions)  # 期望每個類的 recall 相同
    print(f"  如果按訓練集比例預測: UAR ≈ {proportional_uar:.2f}%")

print("\n" + "="*60)
print("對比實際 UAR")
print("="*60)

actual_results = {
    1: 75.74,
    2: 73.95,
    3: 73.93,
    4: 65.99,
    5: 67.61
}

print(f"\n{'Fold':<6} {'最小類':<8} {'不平衡':<10} {'理論UAR':<12} {'實際UAR':<12} {'差距'}")
print("-" * 65)

for fold_num in range(1, 6):
    train_data, test_data = folds[fold_num - 1]
    test_emotions = Counter([item['emotion_label'] for item in test_data])
    
    min_count = min(test_emotions.values())
    max_count = max(test_emotions.values())
    imbalance = max_count / min_count
    
    # 理論最低 UAR（只預測最大類）
    theoretical_uar = 25.0  # 4個類，只有1個是100%
    
    actual_uar = actual_results[fold_num]
    diff = actual_uar - theoretical_uar
    
    print(f"{fold_num:<6} {min_count:<8} {imbalance:<10.2f} {theoretical_uar:<12.2f} {actual_uar:<12.2f} {diff:+.2f}%")

print("\n" + "="*60)
print("關鍵發現")
print("="*60)

# 找出異常的 fold
for fold_num in range(1, 6):
    train_data, test_data = folds[fold_num - 1]
    test_emotions = Counter([item['emotion_label'] for item in test_data])
    
    min_count = min(test_emotions.values())
    max_count = max(test_emotions.values())
    imbalance = max_count / min_count
    actual_uar = actual_results[fold_num]
    
    print(f"\nFold {fold_num}:")
    
    # 檢查 1: 最小類太少
    if min_count < 150:
        print(f"  ⚠️  最小類只有 {min_count} 樣本，容易被忽略")
    
    # 檢查 2: 不平衡但 UAR 高
    if imbalance > 4.0 and actual_uar > 70:
        print(f"  ⚠️  不平衡 {imbalance:.2f}:1 但 UAR 達 {actual_uar:.2f}%")
        print(f"      這可能意味著:")
        print(f"      1. 最小類的樣本特別容易識別")
        print(f"      2. 或者模型在這個 fold 上過擬合")
    
    # 檢查 3: UAR 遠低於預期
    if actual_uar < 70:
        print(f"  ❌ UAR 只有 {actual_uar:.2f}%，明顯低於其他 fold")
        print(f"      這個 fold 的訓練有問題")

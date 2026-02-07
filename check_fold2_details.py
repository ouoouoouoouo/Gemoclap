# check_fold2_details.py
from config import Config
from dataset import load_iemocap_data, create_fold_splits
from collections import Counter

data = load_iemocap_data(Config.IEMOCAP_PATH)
folds = create_fold_splits(data, n_folds=5)

train_data, test_data = folds[1]  # Fold 2

print("="*60)
print("Fold 2 詳細分析")
print("="*60)

test_emotions = Counter([item['emotion_label'] for item in test_data])
emotion_names = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}

print(f"\n測試集詳細分布 ({len(test_data)} 樣本):")
for emotion_id in sorted(test_emotions.keys()):
    emotion_name = emotion_names[emotion_id]
    count = test_emotions[emotion_id]
    percentage = count / len(test_data) * 100
    
    marker = ""
    if count == min(test_emotions.values()):
        marker = "← 最小類！"
    elif count == max(test_emotions.values()):
        marker = "← 最大類"
    
    print(f"  {emotion_name}: {count:4d} ({percentage:5.1f}%) {marker}")

print(f"\n如果 Fold 2 達到 73.95% UAR，每個類的 recall 應該是:")

# 反推每個類的 recall
# 假設 UAR = 73.95%，每個類的平均 recall = 73.95%
# 但考慮到不平衡，我們看看可能的組合

target_uar = 73.95

print(f"\n假設最小類 recall = X%，其他三個類平均 recall = Y%")
print(f"則: (X + 3*Y) / 4 = {target_uar}%")
print(f"\n可能的組合:")

for min_class_recall in [50, 60, 70, 80, 90]:
    other_avg = (target_uar * 4 - min_class_recall) / 3
    print(f"  最小類: {min_class_recall}%, 其他類平均: {other_avg:.1f}%")

# 檢查訓練集
train_emotions = Counter([item['emotion_label'] for item in train_data])
print(f"\n訓練集分布 ({len(train_data)} 樣本):")
for emotion_id in sorted(train_emotions.keys()):
    emotion_name = emotion_names[emotion_id]
    count = train_emotions[emotion_id]
    percentage = count / len(train_data) * 100
    print(f"  {emotion_name}: {count:4d} ({percentage:5.1f}%)")

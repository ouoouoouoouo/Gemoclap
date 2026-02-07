"""
GEmo-CLAP 訓練腳本
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR  # ⭐ 添加這行
from transformers import AutoTokenizer
from tqdm import tqdm
import json

from config import Config
from dataset import (
    IEMOCAPDataset, 
    load_iemocap_data, 
    create_fold_splits,
    collate_fn
)
from models import create_model
from evaluate import evaluate_model


def set_seed(seed):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, device, model_type):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS
    
    pbar = tqdm(dataloader, desc="訓練中")
    optimizer.zero_grad()
    
    # 累積多個 mini-batch 的 embedding，形成更大的對比 batch
    accum_audio_embeddings = []
    accum_text_embeddings = []
    accum_emotion_labels = []
    accum_gender_labels = []
    
    for batch_idx, batch in enumerate(pbar):
        # 移動數據到設備
        audio_input = {k: v.to(device) for k, v in batch["audio_input"].items()}
        text_input = {k: v.to(device) for k, v in batch["text_input"].items()}
        emotion_labels = batch["emotion_label"].to(device)
        gender_labels = batch["gender_label"].to(device)
        
        # 前向傳播
        audio_embeddings, text_embeddings = model(audio_input, text_input)

        # 累積 embeddings 與 labels（保持計算圖，形成更大對比 batch）
        accum_audio_embeddings.append(audio_embeddings)
        accum_text_embeddings.append(text_embeddings)
        accum_emotion_labels.append(emotion_labels)
        accum_gender_labels.append(gender_labels)
        
        is_update_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_step = (batch_idx + 1) == len(dataloader)
        
        if is_update_step or is_last_step:
            # 合併為大 batch
            audio_batch = torch.cat(accum_audio_embeddings, dim=0)
            text_batch = torch.cat(accum_text_embeddings, dim=0)
            emotion_batch = torch.cat(accum_emotion_labels, dim=0)
            gender_batch = torch.cat(accum_gender_labels, dim=0)
            
            # 處理 DataParallel 包裝的模型
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # 計算損失（對比學習需要大 batch）
            if model_type == "emo_clap":
                loss = actual_model.compute_contrastive_loss(
                    audio_batch, text_batch, emotion_batch
                )
            elif model_type == "ml_gemo_clap":
                loss = actual_model.compute_multitask_loss(
                    audio_batch, text_batch,
                    emotion_batch, gender_batch
                )
            elif model_type == "sl_gemo_clap":
                loss = actual_model.compute_soft_label_loss(
                    audio_batch, text_batch,
                    emotion_batch, gender_batch
                )
            else:
                raise ValueError(f"未知的模型類型: {model_type}")
            
            # 反向傳播與更新
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": f"{(batch_idx + 1) // accumulation_steps}/{len(dataloader) // accumulation_steps}"
            })
            
            # 清空累積
            accum_audio_embeddings.clear()
            accum_text_embeddings.clear()
            accum_emotion_labels.clear()
            accum_gender_labels.clear()
    
    # 平均 loss 以更新步數計
    update_steps = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
    avg_loss = total_loss / max(update_steps, 1)
    return avg_loss


def train_fold(model, train_loader, val_loader, optimizer, scheduler, device,  # ⭐ 添加 scheduler
               num_epochs, model_type, save_dir, fold_num, patience=80):
    """
    訓練一個 fold
    
    Args:
        scheduler: 學習率調度器
        patience: Early stopping 的耐心值
    """
    best_uar = 0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "val_war": [],
        "val_uar": [],
        "learning_rate": []  # ⭐ 記錄學習率
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Fold {fold_num} - Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # ⭐ 打印當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"當前學習率: {current_lr:.2e}")
        
        # 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, device, model_type)
        print(f"訓練損失: {train_loss:.4f}")
        
        # 驗證
        val_metrics = evaluate_model(model, val_loader, device)
        val_war = val_metrics["WAR"]
        val_uar = val_metrics["UAR"]
        
        print(f"驗證 WAR: {val_war:.2f}%")
        print(f"驗證 UAR: {val_uar:.2f}%")
        
        # 保存歷史
        history["train_loss"].append(train_loss)
        history["val_war"].append(val_war)
        history["val_uar"].append(val_uar)
        history["learning_rate"].append(current_lr)  # ⭐ 記錄學習率
        
        # ⭐ 更新學習率（在 epoch 結束時）
        scheduler.step()
        
        # 保存最佳模型
        if val_uar > best_uar:
            best_uar = val_uar
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"fold_{fold_num}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),  # ⭐ 保存 scheduler
                    "uar": best_uar,
                    "war": val_war
                }, save_path)
                print(f"✅ 保存最佳模型到 {save_path} (UAR: {best_uar:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"⚠️  UAR 沒有提升 ({epochs_no_improve}/{patience})")
            
            # Early Stopping
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early Stopping: UAR 在 {patience} 個 epoch 內沒有提升")
                print(f"   最佳 UAR: {best_uar:.2f}% (Epoch {best_epoch})")
                break
    
    print(f"\nFold {fold_num} 最佳結果: UAR={best_uar:.2f}% (Epoch {best_epoch})")
    
    return best_uar, history


def main(args):
    # 設置隨機種子
    set_seed(Config.SEED)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"設備配置")
    print(f"{'='*60}")
    print(f"使用設備: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU 型號: {gpu_name}")
        print(f"GPU 顯存: {gpu_memory:.2f} GB")
        print(f"GPU 數量: {torch.cuda.device_count()}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"✅ 將使用 GPU 加速訓練")
    else:
        print(f"⚠️  警告: CUDA 不可用")
    print(f"{'='*60}\n")
    
    # 載入數據
    print("\n載入 IEMOCAP 數據集...")
    data_list = load_iemocap_data(Config.IEMOCAP_PATH)
    
    # 創建 fold 分割
    print("\n創建 5-fold 交叉驗證分割...")
    folds = create_fold_splits(data_list, n_folds=Config.N_FOLDS)
    
    # 初始化處理器
    print("\n載入預訓練模型和處理器...")
    text_processor = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
    
    from transformers import Wav2Vec2FeatureExtractor
    audio_encoder_path = Config.AUDIO_ENCODERS[args.audio_encoder]
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
    
    # 存儲每個 fold 的結果
    fold_results = []
    
    # 訓練每個 fold
    for fold_num, (train_data, test_data) in enumerate(folds, 1):
        print(f"\n{'#'*60}")
        print(f"開始訓練 Fold {fold_num}/{Config.N_FOLDS}")
        print(f"{'#'*60}")
        
        # 創建數據集
        train_dataset = IEMOCAPDataset(
            train_data, 
            audio_processor=audio_processor,
            text_processor=text_processor,
            mode='train'
        )
        test_dataset = IEMOCAPDataset(
            test_data,
            audio_processor=audio_processor,
            text_processor=text_processor,
            mode='test'
        )
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 創建模型
        model = create_model(
            model_type=args.model_type,
            text_encoder=Config.TEXT_ENCODER,
            audio_encoder=args.audio_encoder,
            embedding_dim=Config.EMBEDDING_DIM,
            alpha_e=Config.ALPHA_E if args.model_type != "emo_clap" else None
        )
        
        # 多 GPU 支持
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 張 GPU")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # ⭐ 創建優化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE
        )
        
        # ⭐ 創建學習率調度器
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,      # 總 epoch 數
            eta_min=1e-6            # 最小學習率
        )
        
        print(f"\n📊 訓練設置:")
        print(f"  初始學習率: {Config.LEARNING_RATE:.2e}")
        print(f"  最小學習率: 1e-6")
        print(f"  調度器: CosineAnnealingLR")
        print(f"  總 epochs: {args.epochs}")
        
        # ⭐ 訓練（傳入 scheduler）
        best_uar, history = train_fold(
            model, train_loader, test_loader, optimizer, scheduler, device,  # 傳入 scheduler
            num_epochs=args.epochs,
            model_type=args.model_type,
            save_dir=args.save_dir,
            fold_num=fold_num
        )
        
        fold_results.append({
            "fold": fold_num,
            "best_uar": best_uar,
            "history": history
        })
    
    # 計算平均結果
    avg_uar = np.mean([r["best_uar"] for r in fold_results])
    std_uar = np.std([r["best_uar"] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"5-Fold 交叉驗證結果")
    print(f"{'='*60}")
    print(f"平均 UAR: {avg_uar:.2f}% (±{std_uar:.2f}%)")
    
    for result in fold_results:
        print(f"  Fold {result['fold']}: UAR={result['best_uar']:.2f}%")
    
    # 保存結果
    if args.save_dir:
        results_path = os.path.join(args.save_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model_type": args.model_type,
                "audio_encoder": args.audio_encoder,
                "avg_uar": float(avg_uar),
                "std_uar": float(std_uar),
                "fold_results": fold_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n結果已保存到 {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 GEmo-CLAP 模型")
    
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
        choices=["wav2vec2", "hubert", "wavlm", "wavlm-base", "data2vec"],  # ⭐ 添加 wavlm
        help="音頻編碼器類型"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="訓練輪數"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="模型保存目錄"
    )
    
    args = parser.parse_args()
    
    print(f"\n訓練配置:")
    print(f"  模型類型: {args.model_type}")
    print(f"  音頻編碼器: {args.audio_encoder}")
    print(f"  訓練輪數: {args.epochs}")
    print(f"  批次大小: {Config.BATCH_SIZE}")
    print(f"  學習率: {Config.LEARNING_RATE}")
    
    main(args)
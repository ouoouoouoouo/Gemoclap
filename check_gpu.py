"""
檢查 GPU 和 CUDA 環境
"""

import sys

print("="*60)
print("GPU 環境檢測")
print("="*60)

# 檢查 PyTorch
try:
    import torch
    print(f"\n✅ PyTorch 已安裝")
    print(f"   版本: {torch.__version__}")
except ImportError:
    print("\n❌ PyTorch 未安裝")
    print("   請運行: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# 檢查 CUDA
print(f"\n{'='*60}")
print("CUDA 狀態檢查")
print("="*60)

cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✅ CUDA 可用")
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
    
    # GPU 資訊
    gpu_count = torch.cuda.device_count()
    print(f"\n✅ 檢測到 {gpu_count} 個 GPU:")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"\n   GPU {i}: {gpu_name}")
        print(f"   顯存: {gpu_memory:.2f} GB")
        
        # 測試 GPU 記憶體
        if cuda_available:
            torch.cuda.set_device(i)
            # 嘗試分配一些記憶體
            try:
                test_tensor = torch.zeros(1000, 1000).cuda()
                current_memory = torch.cuda.memory_allocated(i) / (1024**3)
                reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   當前使用: {current_memory:.2f} GB")
                print(f"   已預留: {reserved_memory:.2f} GB")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   ⚠️  記憶體測試失敗: {e}")
    
    # 推薦設置
    print(f"\n{'='*60}")
    print("訓練建議")
    print("="*60)
    
    if gpu_memory >= 24:
        print("✅ 顯存充足 (≥24GB)")
        print("   推薦配置: BATCH_SIZE=32, 使用 WavLM")
    elif gpu_memory >= 12:
        print("⚠️  顯存中等 (12-24GB)")
        print("   推薦配置: BATCH_SIZE=16, 使用 WavLM")
        print("   或者: BATCH_SIZE=32, 使用 Wav2Vec2")
    else:
        print("❌ 顯存不足 (<12GB)")
        print("   推薦配置: BATCH_SIZE=8, 使用 Wav2Vec2")
        print("   警告: 可能無法訓練大型模型")

else:
    print("❌ CUDA 不可用")
    print("\n可能的原因:")
    print("1. 沒有 NVIDIA GPU")
    print("2. NVIDIA 驅動未安裝")
    print("3. PyTorch 安裝的是 CPU 版本")
    print("\n解決方案:")
    print("- 確認已安裝 NVIDIA GPU 驅動")
    print("- 重新安裝 PyTorch GPU 版本:")
    print("  pip uninstall torch torchaudio")
    print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")

# 檢查其他依賴
print(f"\n{'='*60}")
print("其他依賴檢查")
print("="*60)

dependencies = {
    "transformers": "Hugging Face Transformers",
    "torchaudio": "TorchAudio",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "tqdm": "TQDM",
    "librosa": "Librosa",
    "soundfile": "SoundFile"
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name}")
        missing.append(module)

if missing:
    print(f"\n⚠️  缺少依賴: {', '.join(missing)}")
    print("請運行: pip install -r requirements.txt")
else:
    print("\n✅ 所有依賴已安裝完成")

# 最終建議
print(f"\n{'='*60}")
print("環境狀態總結")
print("="*60)

if cuda_available and not missing:
    print("✅ 環境配置完成，可以開始訓練！")
    print("\n建議運行:")
    print("python test_data_loading.py  # 測試數據載入")
    print("python train.py --model_type sl_gemo_clap --audio_encoder wavlm --epochs 10  # 快速測試")
elif cuda_available and missing:
    print("⚠️  GPU 可用但缺少依賴套件")
    print("請先安裝缺少的套件: pip install -r requirements.txt")
elif not cuda_available and not missing:
    print("⚠️  依賴完整但 GPU 不可用")
    print("訓練將使用 CPU，速度會非常慢（不推薦）")
else:
    print("❌ 環境未配置完成")
    print("請先解決上述問題")

print("="*60)


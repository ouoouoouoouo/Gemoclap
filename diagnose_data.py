"""
診斷 IEMOCAP 數據集載入問題
"""

import os
from config import Config

def diagnose_iemocap():
    """診斷 IEMOCAP 數據集"""
    
    print("="*60)
    print("IEMOCAP 數據集診斷")
    print("="*60)
    
    iemocap_path = Config.IEMOCAP_PATH
    print(f"\n1. 檢查數據集路徑:")
    print(f"   配置路徑: {iemocap_path}")
    print(f"   絕對路徑: {os.path.abspath(iemocap_path)}")
    print(f"   路徑存在: {os.path.exists(iemocap_path)}")
    
    if not os.path.exists(iemocap_path):
        print("\n❌ 錯誤: IEMOCAP 路徑不存在！")
        print("\n可能的解決方案:")
        print("1. 檢查數據集是否已上傳")
        print("2. 修改 config.py 中的 IEMOCAP_PATH")
        print(f"   當前: IEMOCAP_PATH = '{iemocap_path}'")
        print(f"   可能需要: IEMOCAP_PATH = '~/IEMOCAP_full_release' 或完整路徑")
        return
    
    print(f"   ✅ 路徑存在")
    
    # 檢查 Sessions
    print(f"\n2. 檢查 Sessions:")
    for session_num in Config.SESSIONS:
        session_path = os.path.join(iemocap_path, f"Session{session_num}")
        exists = os.path.exists(session_path)
        print(f"   Session{session_num}: {'✅' if exists else '❌'} {session_path}")
        
        if exists:
            # 檢查子目錄
            dialog_path = os.path.join(session_path, "dialog")
            sentences_path = os.path.join(session_path, "sentences")
            print(f"     - dialog/: {'✅' if os.path.exists(dialog_path) else '❌'}")
            print(f"     - sentences/: {'✅' if os.path.exists(sentences_path) else '❌'}")
    
    # 檢查標註文件
    print(f"\n3. 檢查標註文件 (Session1 為例):")
    session1_path = os.path.join(iemocap_path, "Session1")
    categorical_path = os.path.join(session1_path, "dialog", "EmoEvaluation", "Categorical")
    
    print(f"   路徑: {categorical_path}")
    print(f"   存在: {os.path.exists(categorical_path)}")
    
    if os.path.exists(categorical_path):
        cat_files = [f for f in os.listdir(categorical_path) if f.endswith("_cat.txt")]
        print(f"   ✅ 找到 {len(cat_files)} 個標註文件")
        if cat_files:
            print(f"   示例: {cat_files[0]}")
            
            # 讀取第一個文件的內容
            sample_file = os.path.join(categorical_path, cat_files[0])
            print(f"\n   文件內容示例:")
            with open(sample_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                for line in lines:
                    print(f"     {line.strip()}")
        else:
            print(f"   ❌ 沒有找到 *_cat.txt 文件")
    else:
        print(f"   ❌ 標註文件夾不存在")
    
    # 檢查音頻文件
    print(f"\n4. 檢查音頻文件 (Session1 為例):")
    wav_path = os.path.join(session1_path, "sentences", "wav")
    
    print(f"   路徑: {wav_path}")
    print(f"   存在: {os.path.exists(wav_path)}")
    
    if os.path.exists(wav_path):
        # 列出子文件夾
        subdirs = [d for d in os.listdir(wav_path) if os.path.isdir(os.path.join(wav_path, d))]
        print(f"   ✅ 找到 {len(subdirs)} 個對話文件夾")
        if subdirs:
            print(f"   示例: {subdirs[0]}")
            
            # 檢查第一個子文件夾的 wav 文件
            first_dir = os.path.join(wav_path, subdirs[0])
            wav_files = [f for f in os.listdir(first_dir) if f.endswith('.wav')]
            print(f"   該文件夾中的 wav 文件: {len(wav_files)}")
            if wav_files:
                print(f"   示例: {wav_files[0]}")
        else:
            print(f"   ❌ 沒有找到對話文件夾")
    else:
        print(f"   ❌ 音頻文件夾不存在")
    
    # 嘗試載入數據
    print(f"\n5. 嘗試載入數據:")
    try:
        from dataset import load_iemocap_data
        data_list = load_iemocap_data(iemocap_path, sessions=[1])  # 只載入 Session1
        print(f"   載入的數據數量: {len(data_list)}")
        
        if len(data_list) > 0:
            print(f"   ✅ 數據載入成功！")
            print(f"\n   示例數據:")
            audio_path, emotion, gender = data_list[0]
            print(f"     音頻: {audio_path}")
            print(f"     情緒: {emotion}")
            print(f"     性別: {gender}")
            print(f"     文件存在: {os.path.exists(audio_path)}")
        else:
            print(f"   ❌ 數據載入失敗，沒有載入任何數據")
            print(f"\n   可能的原因:")
            print(f"   1. 標註文件格式不正確")
            print(f"   2. 情緒標籤不匹配")
            print(f"   3. 音頻文件路徑構建錯誤")
            
    except Exception as e:
        print(f"   ❌ 載入數據時出錯: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("診斷完成")
    print("="*60)

if __name__ == "__main__":
    diagnose_iemocap()



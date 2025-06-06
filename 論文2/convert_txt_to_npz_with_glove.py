import numpy as np
import os

# 設定路徑
base_path = "Features/resnet101"
feature_file = os.path.join(base_path, "awa2-features.txt")
label_file = os.path.join(base_path, "awa2-labels.txt")
filename_file = os.path.join(base_path, "awa2-filenames.txt")
glove_file = "glove.6B.50d.txt"  # 放在同一層

# 載入 GloVe 詞向量
print("🔄 載入 GloVe...")
glove = {}
with open(glove_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        glove[word] = vec
print(f"✅ GloVe 向量載入，共 {len(glove)} 詞")

# 讀入資料
features = np.loadtxt(feature_file)
labels = np.loadtxt(label_file).astype(int) - 1
with open(filename_file, "r") as f:
    filenames = [line.strip() for line in f]

# 所有圖片對應的類別名稱
all_class_names = [name.split("/")[0] for name in filenames]

# 利用 label 決定 class_name（共 50 類）
class_names = [all_class_names[np.where(labels == i)[0][0]] for i in range(50)]
print(f"✅ 類別共 {len(class_names)} 種")
print("🔎 前 5 類別名稱：", class_names[:5])

# 類別名稱轉成 GloVe 向量
def get_glove_embedding(name):
    words = name.replace("_", " ").split()
    vecs = [glove[w] for w in words if w in glove]
    if not vecs:
        return np.zeros(50, dtype=np.float32)
    return np.mean(vecs, axis=0)

attributes = np.stack([get_glove_embedding(name) for name in class_names])

# 儲存為 .npz
np.savez("awa2_data_glove.npz",
         features=features,
         labels=labels,
         filenames=filenames,
         class_names=class_names,
         attributes=attributes)

print("✅ 轉換完成 → awa2_data_glove.npz")
print("語意向量 shape:", attributes.shape)

import numpy as np
import tensorflow as tf
from seen_expert_model import SeenExpertModel

# 載入資料
data = np.load("awa2_data_glove.npz", allow_pickle=True)
features = data["features"]
labels = data["labels"]
attributes = data["attributes"]
class_names = data["class_names"]

# 分類設定
num_classes = 50
seen_classes = np.arange(40)
unseen_classes = np.arange(40, 50)

# 訓練資料（seen）
train_idx = np.isin(labels, seen_classes)
x_train = features[train_idx]
y_train = labels[train_idx]

# 測試資料（unseen）
test_idx = np.isin(labels, unseen_classes)
x_test = features[test_idx]
y_test = labels[test_idx]

print("✅ 訓練樣本：", x_train.shape)
print("✅ 測試樣本（unseen）：", x_test.shape)

# 模型參數
semantic_dim = attributes.shape[1]
visual_dim = features.shape[1]

# 語意向量
seen_attributes = attributes[seen_classes]       # (40, 50)
unseen_attributes = attributes[unseen_classes]   # (10, 50)

# ===== 模型訓練階段 ===== #
class SemanticWrapper(tf.keras.Model):
    def __init__(self, model, semantic_matrix):
        super().__init__()
        self.model = model
        self.semantic_matrix = tf.convert_to_tensor(semantic_matrix, dtype=tf.float32)

    def call(self, x):
        return self.model(x, self.semantic_matrix)

# 建立 seen 模型
model_train = SeenExpertModel(semantic_dim, visual_dim)
wrapped_train = SemanticWrapper(model_train, seen_attributes)

wrapped_train.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

# 重新編碼 y_train 為 0–39
seen_class_to_index = {cls: i for i, cls in enumerate(seen_classes)}
y_train_remap = np.array([seen_class_to_index[y] for y in y_train])

# 訓練
wrapped_train.fit(x_train, y_train_remap, epochs=5, batch_size=64)

# 建立測試模型
model_test = SeenExpertModel(semantic_dim, visual_dim)
wrapped_test = SemanticWrapper(model_test, unseen_attributes)

# 先初始化 wrapped_test（⚠️ 這一行是關鍵）
wrapped_test(tf.random.normal((1, visual_dim)))

# 複製訓練模型權重
wrapped_test.set_weights(wrapped_train.get_weights())

wrapped_test.compile(optimizer="adam",
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])

# 重新編碼 y_test 為 0–9
unseen_class_to_index = {cls: i for i, cls in enumerate(unseen_classes)}
y_test_remap = np.array([unseen_class_to_index[y] for y in y_test])

# 評估
loss, acc = wrapped_test.evaluate(x_test, y_test_remap)
print(f"🎯 Constrained ZSL 測試準確率（僅預測 unseen 10 類）: {acc:.4f}")


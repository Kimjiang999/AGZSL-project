import numpy as np
import tensorflow as tf
from seen_expert_model import SeenExpertModel

# 載入資料
data = np.load("awa2_data_glove.npz", allow_pickle=True)
features = data["features"]
labels = data["labels"]
attributes = data["attributes"]  # shape = (50, 50)

# 切分資料
split = int(len(features) * 0.8)
x_train, x_test = features[:split], features[split:]
y_train, y_test = labels[:split], labels[split:]

# 模型參數
semantic_dim = attributes.shape[1]
visual_dim = features.shape[1]

# 包裝模型
class SemanticWrapper(tf.keras.Model):
    def __init__(self, model, attribute_matrix):
        super().__init__()
        self.model = model
        self.semantic_matrix = tf.convert_to_tensor(attribute_matrix, dtype=tf.float32)

    def call(self, x):
        return self.model(x, self.semantic_matrix)

model = SeenExpertModel(semantic_dim, visual_dim)
wrapped_model = SemanticWrapper(model, attributes)

wrapped_model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

wrapped_model.fit(x_train, y_train, epochs=5, batch_size=64)

# 測試
loss, acc = wrapped_model.evaluate(x_test, y_test)
print(f"✅ 使用 GloVe 語意向量的測試準確率：{acc:.4f}")
import tensorflow as tf
from seen_expert_model import SeenExpertModel

# 模擬資料參數
batch_size = 8
visual_dim = 2048
semantic_dim = 85
num_classes = 5
epochs = 10

# 建立模擬資料（影像特徵 + 對應標籤）
x_train = tf.random.normal((batch_size * 10, visual_dim))           # 80 筆影像特徵
y_train = tf.random.uniform((batch_size * 10,), minval=0, maxval=num_classes, dtype=tf.int32)

semantic_matrix = tf.random.normal((num_classes, semantic_dim))    # 類別語意向量

# 建立模型
model = SeenExpertModel(semantic_dim, visual_dim)

# 自訂 loss：使用 sparse categorical crossentropy
def loss_fn(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

# 編譯模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=['sparse_categorical_accuracy']
)

# 包裝為 tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(100).batch(batch_size)

# 建立 wrapper 把 semantic_matrix 傳入模型
class SemanticWrapper(tf.keras.Model):
    def __init__(self, model, semantic_matrix):
        super().__init__()
        self.model = model
        self.semantic_matrix = semantic_matrix

    def call(self, x):
        return self.model(x, self.semantic_matrix)

train_model = SemanticWrapper(model, semantic_matrix)

# 編譯再次（因為換了外層 Model）
train_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=['sparse_categorical_accuracy']
)

# 訓練
print("準備開始訓練")
train_model.fit(train_dataset, epochs=epochs)
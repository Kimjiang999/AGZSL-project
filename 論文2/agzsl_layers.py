import tensorflow as tf
from tensorflow.keras import layers

class IASLayer(tf.keras.layers.Layer):
    def __init__(self, semantic_dim):
        super(IASLayer, self).__init__()
        self.attention_layer = layers.Dense(semantic_dim)

    def call(self, visual_feature, semantic_features):
        attn_weights = tf.nn.softmax(self.attention_layer(visual_feature), axis=-1)
        attn_weights = tf.expand_dims(attn_weights, axis=1)
        sem_exp = tf.expand_dims(semantic_features, axis=0)
        weighted_semantics = sem_exp * attn_weights
        output = sem_exp + weighted_semantics
        return output

class S2VLayer(tf.keras.layers.Layer):
    def __init__(self, semantic_dim, visual_dim, hidden_dim=1600):
        super(S2VLayer, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(visual_dim)

    def call(self, semantic_input):
        x = self.fc1(semantic_input)
        x = self.fc2(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        return x

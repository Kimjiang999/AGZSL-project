import tensorflow as tf
from tensorflow.keras import Model
from agzsl_layers import IASLayer, S2VLayer

class SeenExpertModel(Model):
    def __init__(self, semantic_dim, visual_dim, hidden_dim=1600):
        super(SeenExpertModel, self).__init__()
        self.ias = IASLayer(semantic_dim)
        self.s2v = S2VLayer(semantic_dim, visual_dim, hidden_dim)

    def call(self, visual_input, semantic_matrix):
        adapted_sem = self.ias(visual_input, semantic_matrix)
        projected_visuals = self.s2v(adapted_sem)
        norm_visual_input = tf.nn.l2_normalize(visual_input, axis=-1)
        norm_visual_input = tf.expand_dims(norm_visual_input, axis=1)
        cosine_sim = tf.reduce_sum(norm_visual_input * projected_visuals, axis=-1)
        logits = tf.nn.softmax(cosine_sim, axis=-1)
        return logits

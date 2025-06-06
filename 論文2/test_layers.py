import tensorflow as tf
from agzsl_layers import IASLayer, S2VLayer

batch_size = 2
visual_dim = 2048
semantic_dim = 85
num_classes = 5

visual_input = tf.random.normal((batch_size, visual_dim))
semantic_matrix = tf.random.normal((num_classes, semantic_dim))

ias = IASLayer(semantic_dim)
s2v = S2VLayer(semantic_dim, visual_dim)

adapted_sem = ias(visual_input, semantic_matrix)
projected_visuals = s2v(adapted_sem)

print("Adaptive semantics shape:", adapted_sem.shape)
print("Projected visuals shape:", projected_visuals.shape)
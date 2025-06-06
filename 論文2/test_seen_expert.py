import tensorflow as tf
from seen_expert_model import SeenExpertModel

batch_size = 2
visual_dim = 2048
semantic_dim = 85
num_classes = 5

visual_input = tf.random.normal((batch_size, visual_dim))
semantic_matrix = tf.random.normal((num_classes, semantic_dim))

model = SeenExpertModel(semantic_dim, visual_dim)
logits = model(visual_input, semantic_matrix)

print("Logits shape:", logits.shape)
print("Logits:", logits.numpy())
"""
Simple TensorFlow Recommenders example for comparison with TorchRS
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def run_simple_tfrs_test():
    """Run a simple TFRS test with more interactions for better comparison."""
    print("Running simple TensorFlow Recommenders test...")
    
    # Create a dataset with more interactions to match TorchRS test
    user_ids = [str(i) for i in range(10)] * 10  # 100 interactions
    movie_ids = [str(np.random.randint(0, 20)) for _ in range(100)]
    
    # Convert to TensorFlow dataset
    ratings_data = tf.data.Dataset.from_tensor_slices({
        "user_id": user_ids,
        "movie_id": movie_ids
    })
    
    # Get unique IDs
    unique_user_ids = list(set(user_ids))
    unique_movie_ids = list(set(movie_ids))
    
    # Create vocabularies
    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(unique_user_ids)
    
    movie_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_ids_vocabulary.adapt(unique_movie_ids)
    
    # Define simple model
    class SimpleModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.user_model = tf.keras.Sequential([
                user_ids_vocabulary,
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            self.movie_model = tf.keras.Sequential([
                movie_ids_vocabulary,
                tf.keras.layers.Embedding(len(unique_movie_ids) + 1, 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            self.task = tfrs.tasks.Retrieval()
        
        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            movie_embeddings = self.movie_model(features["movie_id"])
            return self.task(user_embeddings, movie_embeddings)
    
    # Create and compile model
    model = SimpleModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    
    # Train
    train_data = ratings_data.shuffle(1000).batch(16)
    history = model.fit(train_data, epochs=10, verbose=0)
    
    # Print results
    train_losses = history.history['loss']
    print("Training losses:")
    for i, loss in enumerate(train_losses):
        if (i + 1) % 2 == 0:
            print(f"   Epoch {i+1}, Loss: {loss:.4f}")
    
    print("Simple TFRS test completed!")

if __name__ == "__main__":
    run_simple_tfrs_test()
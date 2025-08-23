"""
TensorFlow Recommenders example for comparison with TorchRS
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_standardized_movielens_data():
    """Create a standardized MovieLens-like dataset for consistency testing."""
    # Create a small, fixed dataset that mimics MovieLens
    # User IDs: 0-9 (10 users)
    # Movie IDs: 0-19 (20 movies)
    # Ratings: 1-5
    
    user_ids = []
    movie_ids = []
    ratings = []
    
    # Create a deterministic dataset
    for user_id in range(10):
        # Each user rates 5 random movies
        rated_movies = np.random.choice(20, size=5, replace=False)
        for movie_id in rated_movies:
            rating = np.random.randint(1, 6)  # Rating from 1-5
            user_ids.append(str(user_id))     # TFRS expects strings
            movie_ids.append(str(movie_id))   # TFRS expects strings
            ratings.append(float(rating))
    
    return user_ids, movie_ids, ratings

def run_tfrs_consistency_test():
    """Run a consistency test with TensorFlow Recommenders."""
    print("Running TensorFlow Recommenders consistency test...")
    
    # 1. Load standardized data
    print("1. Loading standardized data...")
    user_ids, movie_ids, ratings = create_standardized_movielens_data()
    
    print(f"   Dataset size: {len(user_ids)} interactions")
    print(f"   Users: {len(set(user_ids))}")
    print(f"   Movies: {len(set(movie_ids))}")
    print(f"   Rating range: {int(np.min(ratings))}-{int(np.max(ratings))}")
    
    # Convert to TensorFlow dataset
    ratings_data = tf.data.Dataset.from_tensor_slices({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings
    })
    
    # Get unique user and movie IDs
    unique_user_ids = list(set(user_ids))
    unique_movie_ids = list(set(movie_ids))
    
    print(f"   Unique users: {len(unique_user_ids)}")
    print(f"   Unique movies: {len(unique_movie_ids)}")
    
    # Create vocabularies
    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(unique_user_ids)
    
    movie_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_ids_vocabulary.adapt(unique_movie_ids)
    
    # 2. Define model
    print("2. Creating model...")
    
    class MovieLensModel(tfrs.Model):
        def __init__(self, user_ids_vocabulary, movie_ids_vocabulary):
            super().__init__()
            self.user_ids_vocabulary = user_ids_vocabulary
            self.movie_ids_vocabulary = movie_ids_vocabulary
            
            # User model: Embedding -> Dense -> ReLU
            self.user_model = tf.keras.Sequential([
                user_ids_vocabulary,
                tf.keras.layers.Embedding(len(user_ids_vocabulary.get_vocabulary()), 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            # Movie model: Embedding -> Dense -> ReLU
            self.movie_model = tf.keras.Sequential([
                movie_ids_vocabulary,
                tf.keras.layers.Embedding(len(movie_ids_vocabulary.get_vocabulary()), 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            # Task
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=tf.data.Dataset.from_tensor_slices(unique_movie_ids)
                        .batch(128)
                        .map(self.movie_model)
                )
            )
        
        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            movie_embeddings = self.movie_model(features["movie_id"])
            return self.task(user_embeddings, movie_embeddings)
    
    # Create model
    model = MovieLensModel(user_ids_vocabulary, movie_ids_vocabulary)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    
    # 3. Train model
    print("3. Training model...")
    train_data = ratings_data.shuffle(1000).batch(16)
    
    # Train for 10 epochs
    history = model.fit(train_data, epochs=10, verbose=0)
    
    # Print training losses
    train_losses = history.history['loss']
    for i, loss in enumerate(train_losses):
        if (i + 1) % 2 == 0:
            print(f"   Epoch {i+1}, Loss: {loss:.4f}")
    
    # 4. Generate recommendations
    print("4. Generating recommendations...")
    
    # Create a retrieval index
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(unique_movie_ids)
            .batch(100)
            .map(lambda x: (x, model.movie_model(x)))
    )
    
    # Get recommendations for user "0"
    _, titles = index(tf.constant(["0"]))
    
    print(f"   Top 5 recommendations for user 0:")
    for i, movie in enumerate(titles[0, :5]):
        # Convert from tensor to string and remove b'' prefix
        movie_id = movie.numpy().decode('utf-8')
        print(f"     {i+1}. Movie {movie_id}")
    
    # 5. Return results for comparison
    results = {
        'train_losses': train_losses,
        'recommendations': [title.numpy().decode('utf-8') for title in titles[0, :5]]
    }
    
    print("\nTensorFlow Recommenders consistency test completed!")
    return results

if __name__ == "__main__":
    results = run_tfrs_consistency_test()
    
    # Print results in a format that could be compared
    print("\n=== TFRS RESULTS SUMMARY ===")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print("Top recommendations for user 0:")
    for i, movie in enumerate(results['recommendations']):
        print(f"  {i+1}. Movie {movie}")
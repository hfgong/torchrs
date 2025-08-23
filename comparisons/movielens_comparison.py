"""
Compare recommendations for specific users using the MovieLens dataset
with both TensorFlow Recommenders and TorchRS
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

def load_movielens_data():
    """Load the MovieLens 100K dataset."""
    print("Loading MovieLens 100K dataset...")
    
    # Load ratings data
    ratings = tfds.load("movielens/100k-ratings", split="train")
    
    # Get movie data
    movies = tfds.load("movielens/100k-movies", split="train")
    
    # Convert to numpy for easier handling
    ratings_numpy = []
    for record in ratings.take(1000):  # Take a subset for faster processing
        ratings_numpy.append({
            "user_id": record["user_id"].numpy().decode('utf-8'),
            "movie_id": record["movie_id"].numpy().decode('utf-8'),
            "user_rating": float(record["user_rating"].numpy())
        })
    
    movies_numpy = {}
    for record in movies:
        movie_id = record["movie_id"].numpy().decode('utf-8')
        movie_title = record["movie_title"].numpy().decode('utf-8')
        movies_numpy[movie_id] = movie_title
    
    print(f"Loaded {len(ratings_numpy)} ratings and {len(movies_numpy)} movies")
    return ratings_numpy, movies_numpy

def run_tfrs_model(ratings_data, movies_data):
    """Run TensorFlow Recommenders model."""
    print("\n=== Running TensorFlow Recommenders ===")
    
    # Convert to TensorFlow dataset
    ratings_tf = tf.data.Dataset.from_tensor_slices({
        "user_id": [r["user_id"] for r in ratings_data],
        "movie_id": [r["movie_id"] for r in ratings_data],
        "user_rating": [r["user_rating"] for r in ratings_data]
    })
    
    # Get unique user and movie IDs
    unique_user_ids = list(set([r["user_id"] for r in ratings_data]))
    unique_movie_ids = list(set([r["movie_id"] for r in ratings_data]))
    
    print(f"Unique users: {len(unique_user_ids)}, Unique movies: {len(unique_movie_ids)}")
    
    # Create vocabularies
    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(unique_user_ids)
    
    movie_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_ids_vocabulary.adapt(unique_movie_ids)
    
    # Define model
    class MovieLensModel(tfrs.Model):
        def __init__(self, user_ids_vocabulary, movie_ids_vocabulary):
            super().__init__()
            self.user_ids_vocabulary = user_ids_vocabulary
            self.movie_ids_vocabulary = movie_ids_vocabulary
            
            # User model
            self.user_model = tf.keras.Sequential([
                user_ids_vocabulary,
                tf.keras.layers.Embedding(len(user_ids_vocabulary.get_vocabulary()), 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            # Movie model
            self.movie_model = tf.keras.Sequential([
                movie_ids_vocabulary,
                tf.keras.layers.Embedding(len(movie_ids_vocabulary.get_vocabulary()), 32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
            
            # Task
            self.task = tfrs.tasks.Retrieval()
        
        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            movie_embeddings = self.movie_model(features["movie_id"])
            return self.task(user_embeddings, movie_embeddings)
    
    # Create and train model
    model = MovieLensModel(user_ids_vocabulary, movie_ids_vocabulary)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    
    # Train
    train_data = ratings_tf.shuffle(1000).batch(32)
    history = model.fit(train_data, epochs=5, verbose=0)
    
    # Print training losses
    train_losses = history.history['loss']
    print(f"TFRS Training losses: {[f'{loss:.4f}' for loss in train_losses]}")
    
    # Create retrieval index
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(unique_movie_ids)
            .batch(100)
            .map(lambda x: (x, model.movie_model(x)))
    )
    
    return model, index, movies_data

def run_torchrs_model(ratings_data, movies_data):
    """Run TorchRS model."""
    print("\n=== Running TorchRS ===")
    
    # Convert data to integer IDs for PyTorch
    user_id_map = {uid: i for i, uid in enumerate(set([r["user_id"] for r in ratings_data]))}
    movie_id_map = {mid: i for i, mid in enumerate(set([r["movie_id"] for r in ratings_data]))}
    movie_id_reverse_map = {i: mid for mid, i in movie_id_map.items()}
    
    # Create PyTorch dataset
    user_ids = [user_id_map[r["user_id"]] for r in ratings_data]
    movie_ids = [movie_id_map[r["movie_id"]] for r in ratings_data]
    ratings = [r["user_rating"] for r in ratings_data]
    
    dataset = data.RecommendationDataset(user_ids, movie_ids, ratings)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Unique users: {len(user_id_map)}, Unique movies: {len(movie_id_map)}")
    
    # Create models
    user_model = nn.Sequential(
        nn.Embedding(len(user_id_map), 32),
        nn.Linear(32, 16),
        nn.ReLU()
    )
    
    movie_model = nn.Sequential(
        nn.Embedding(len(movie_id_map), 32),
        nn.Linear(32, 16),
        nn.ReLU()
    )
    
    # Create task and model
    task = tasks.Retrieval(metrics=[metrics.FactorizedTopK(k=10)])
    model = models.RetrievalModel(user_model=user_model, item_model=movie_model, task=task)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    train_losses = []
    for epoch in range(5):
        epoch_loss = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            user_ids_batch = batch['user_id']
            movie_ids_batch = batch['item_id']
            loss = model.compute_loss(user_ids_batch, movie_ids_batch, movie_ids_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
    
    print(f"TorchRS Training losses: {[f'{loss:.4f}' for loss in train_losses]}")
    
    return model, user_id_map, movie_id_map, movie_id_reverse_map, movies_data

def compare_recommendations(tfrs_model, tfrs_index, torchrs_model, user_id_maps, movie_id_maps, movies_data):
    """Compare recommendations for specific users."""
    print("\n=== Comparing Recommendations ===")
    
    torchrs_user_id_map, torchrs_movie_id_map, torchrs_movie_id_reverse_map = user_id_maps
    tfrs_user_id_map, tfrs_movie_id_map = movie_id_maps
    
    # Select a few users to compare (let's use the first few)
    sample_users = list(torchrs_user_id_map.keys())[:3]
    
    for user_id in sample_users:
        print(f"\n--- Recommendations for User {user_id} ---")
        
        # Get TFRS recommendations
        try:
            _, tfrs_movie_ids = tfrs_index(tf.constant([user_id]))
            tfrs_recs = [movie_id.decode('utf-8') for movie_id in tfrs_movie_ids[0, :5].numpy()]
            print(f"TFRS Top 5: {tfrs_recs}")
            
            # Get movie titles
            tfrs_titles = [movies_data.get(mid, f"Unknown ({mid})") for mid in tfrs_recs]
            print(f"TFRS Titles: {tfrs_titles}")
        except Exception as e:
            print(f"TFRS Error: {e}")
        
        # Get TorchRS recommendations
        try:
            torchrs_user_idx = torchrs_user_id_map[user_id]
            candidate_movies = torch.arange(0, len(torchrs_movie_id_map))
            
            model.eval()
            with torch.no_grad():
                user_input = torch.tensor([torchrs_user_idx])
                top_movies, top_scores = model.recommend(user_input, candidate_movies, k=5)
                
                torchrs_recs = [torchrs_movie_id_reverse_map[idx.item()] for idx in top_movies[0]]
                print(f"TorchRS Top 5: {torchrs_recs}")
                
                # Get movie titles
                torchrs_titles = [movies_data.get(mid, f"Unknown ({mid})") for mid in torchrs_recs]
                print(f"TorchRS Titles: {torchrs_titles}")
        except Exception as e:
            print(f"TorchRS Error: {e}")

def main():
    """Main function to run the comparison."""
    try:
        # Load data
        ratings_data, movies_data = load_movielens_data()
        
        # Run TFRS model
        tfrs_model, tfrs_index, movies_data = run_tfrs_model(ratings_data, movies_data)
        
        # Run TorchRS model
        torchrs_model, torchrs_user_id_map, torchrs_movie_id_map, torchrs_movie_id_reverse_map, movies_data = run_torchrs_model(ratings_data, movies_data)
        
        # Compare recommendations
        user_id_maps = (torchrs_user_id_map, torchrs_movie_id_map, torchrs_movie_id_reverse_map)
        movie_id_maps = (torchrs_user_id_map, torchrs_movie_id_map)  # Simplified for this example
        compare_recommendations(tfrs_model, tfrs_index, torchrs_model, user_id_maps, movie_id_maps, movies_data)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
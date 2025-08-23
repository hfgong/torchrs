import torch
import sys
import os

# Add the torchrs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Run all tests for the torchrs library."""
    print("Running all tests...")
    
    # Import and run model tests
    print("\n1. Running model tests...")
    try:
        from tests.test_models import test_embedding_creation, test_retrieval_model, test_factorized_topk
        test_embedding_creation()
        test_retrieval_model()
        test_factorized_topk()
        print("   Model tests passed!")
    except Exception as e:
        print(f"   Model tests failed: {e}")
        return False
    
    # Import and run task tests
    print("\n2. Running task tests...")
    try:
        from tests.test_tasks import test_retrieval_task, test_ranking_task
        test_retrieval_task()
        test_ranking_task()
        print("   Task tests passed!")
    except Exception as e:
        print(f"   Task tests failed: {e}")
        return False
    
    # Import and run data tests
    print("\n3. Running data tests...")
    try:
        from tests.test_data import test_recommendation_dataset, test_negative_sampling
        test_recommendation_dataset()
        test_negative_sampling()
        print("   Data tests passed!")
    except Exception as e:
        print(f"   Data tests failed: {e}")
        return False
    
    # Import and run end-to-end test
    print("\n4. Running end-to-end MovieLens test...")
    try:
        from tests.test_movielens_end_to_end import test_movielens_end_to_end
        test_movielens_end_to_end()
        print("   End-to-end test passed!")
    except Exception as e:
        print(f"   End-to-end test failed: {e}")
        return False
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
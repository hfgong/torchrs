import sys
import os

# Add the torchrs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_examples():
    """Run all examples to demonstrate the torchrs library."""
    print("Running torchrs examples...")
    
    # Run retrieval example
    print("\n1. Running MovieLens Retrieval Example...")
    try:
        from examples.movielens_retrieval import main as retrieval_main
        retrieval_main()
        print("   Retrieval example completed successfully!")
    except Exception as e:
        print(f"   Retrieval example failed: {e}")
        return False
    
    # Run ranking example
    print("\n2. Running MovieLens Ranking Example...")
    try:
        from examples.movielens_ranking import main as ranking_main
        ranking_main()
        print("   Ranking example completed successfully!")
    except Exception as e:
        print(f"   Ranking example failed: {e}")
        return False
    
    print("\nAll examples completed successfully!")
    return True

if __name__ == "__main__":
    success = run_examples()
    if not success:
        sys.exit(1)
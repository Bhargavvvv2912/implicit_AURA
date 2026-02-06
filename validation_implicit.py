import sys
import os

# We set this to avoid issues with multi-threading during smoke tests in CI
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def test_implicit_als_functionality():
    print("--- Implicit (ICDM) ALS Functional Verification ---")
    
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        import implicit
        from implicit.als import AlternatingLeastSquares
        
        print(f"--> Libraries imported successfully. Implicit version: {implicit.__version__}")

        # 1. Create a dummy sparse user-item interaction matrix
        # 10 users, 10 items, random ratings
        print("--> Generating synthetic sparse matrix...")
        counts = csr_matrix(np.random.randint(0, 5, size=(10, 10)).astype(np.float32))

        # 2. Initialize the ALS model
        # We use a small factor size for a fast smoke test
        print("--> Initializing ALS Model (Triggers C++/Cython bindings)...")
        model = AlternatingLeastSquares(factors=8, iterations=3, regularization=0.01)

        # 3. Train the model
        # This is where the heavy C++ math kernels are executed
        print("--> Training model...")
        model.fit(counts)

        # 4. Generate recommendations
        print("--> Testing recommendation generation...")
        user_id = 0
        ids, scores = model.recommend(user_id, counts[user_id])

        if len(ids) > 0:
            print(f"    [✓] Model trained. Top recommendation ID: {ids[0]}")
            print("--- SMOKE TEST PASSED ---")
        else:
            print("CRITICAL: No recommendations generated.")
            sys.exit(1)

    except ImportError as ie:
        print(f"CRITICAL BINARY ERROR: {str(ie)}")
        print("Likely caused by NumPy/SciPy ABI mismatch in C++ extensions.")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        # Catch specific NumPy 2.0 AttributeError drifts
        if "numpy" in str(e).lower():
             print("Detected potential NumPy 2.x API conflict.")
        sys.exit(1)

if __name__ == "__main__":
    test_implicit_als_functionality()
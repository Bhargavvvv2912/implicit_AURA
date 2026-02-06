import sys
import os
import glob

def test_implicit_als_functionality():
    print("--- Implicit (ICDM) ALS Functional Verification ---")
    
    # Check for compiled extensions in the local directory
    # Scikit-build/CMake often puts them in implicit/cpu/
    extension_found = glob.glob("implicit/cpu/_als*.so") + glob.glob("implicit/cpu/_als*.pyd")
    if not extension_found:
        print("CRITICAL: Compiled C++ extensions (_als.so) not found in implicit/cpu/")
        print("This usually means the 'pip install -e .' did not trigger a full build.")
        sys.exit(1)

    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        import implicit
        from implicit.als import AlternatingLeastSquares
        
        print(f"--> Libraries imported successfully. Implicit version: {implicit.__version__}")

        # 1. Create a dummy sparse matrix
        counts = csr_matrix(np.random.randint(0, 5, size=(10, 10)).astype(np.float32))

        # 2. Initialize and Train
        # Use 'use_gpu=False' to ensure we only test the CPU C++ kernels in CI
        model = AlternatingLeastSquares(factors=8, iterations=3, use_gpu=False)
        print("--> Training model (Triggers C++/Cython bindings)...")
        model.fit(counts)

        # 3. Generate recommendations
        user_id = 0
        ids, scores = model.recommend(user_id, counts[user_id])

        if len(ids) > 0:
            print(f"    [✓] Model trained. Top recommendation ID: {ids[0]}")
            print("--- SMOKE TEST PASSED ---")
        else:
            sys.exit(1)

    except ImportError as ie:
        print(f"CRITICAL IMPORT ERROR: {str(ie)}")
        print("The C++ extensions were found but could not be loaded (likely ABI mismatch).")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_implicit_als_functionality()
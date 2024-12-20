# Efficient Sparse Triangular Solver Using Triton

This project provides a GPU-accelerated implementation of a sparse triangular solver, leveraging the Triton framework for high-performance kernel development. The solver is designed to handle sparse matrices efficiently and is particularly effective for irregular sparsity patterns.

## Features
- Implements a sparse triangular solver using the Triton framework.
- Handles matrices in Compressed Sparse Row (CSR) format.
- Supports custom sparsity patterns and domain-specific optimizations.
- Includes level scheduling to maximize parallel efficiency.

## Requirements
To run this project, ensure the following dependencies are installed:

- Python 3.8+
- Triton (>=2.0.0)
- PyTorch (>=1.10)
- NVIDIA GPU with CUDA support

## Installation
Follow these steps to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/sparse-triangular-solver.git
   cd sparse-triangular-solver
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   torch>=1.10
   triton>=2.0
   ```

4. Verify the installation by running a test:
   ```bash
   python test_sptrsv.py
   ```

## Usage
The solver can be integrated into any project that involves sparse triangular matrix computations. Example usage:

```python
from sparse_triangular_solver import sptrsv_csr

# Define your CSR matrix and RHS vector
Lp = [0, 3, 6, 8, 9]
Li = [0, 1, 3, 1, 2, 3, 2, 3, 3]
Lx = [10.0, 2.0, 1.0, 3.0, 7.0, 5.0, 2.0, 8.0, 6.0]
RHS = [15.0, 24.0, 14.0, 8.0]

# Call the solver
solution = sptrsv_csr(Lp, Li, Lx, RHS)
print("Solution:", solution)
```

## Testing
Run the provided test script to verify the implementation:

```bash
python test_sptrsv.py
```

## References
- Triton: [https://github.com/openai/triton](https://github.com/openai/triton)
- cuSPARSE Library: [https://developer.nvidia.com/cusparse](https://developer.nvidia.com/cusparse)
- Mehridehnavi, M., & Chowdhury, R. A. (2021). ParSy: Inspection and Transformation of Sparse Matrix Computations for Parallelism. [Paper](https://www.cs.toronto.edu/~mmehride/papers/Parsy.pdf)
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.

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
- Linux
- Triton (>=2.0.0)
- PyTorch (>=1.10)
- NVIDIA GPU with CUDA support
- NVIDIA Drivers

## Installation
Follow these steps to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/loganleon/triton-example.git
   cd triton-example
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

4. Run the code of the triangular solver:
   ```bash
   python sparse_triangular_solver.py
   ```


## References
- Triton: [https://github.com/openai/triton](https://github.com/openai/triton)
- cuSPARSE Library: [https://developer.nvidia.com/cusparse](https://developer.nvidia.com/cusparse)
- Mehridehnavi, M., & Chowdhury, R. A. (2021). ParSy: Inspection and Transformation of Sparse Matrix Computations for Parallelism. [Paper](https://www.cs.toronto.edu/~mmehride/papers/Parsy.pdf)
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.

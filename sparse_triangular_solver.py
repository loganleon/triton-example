import torch
import triton
import triton.language as tl

@triton.jit
def sptrsv_csr_kernel(Lp, Li, Lx, x, level_set, level_ptr, n, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # Get the program ID in the 1D grid

    # Get the current level
    level_start = tl.load(level_ptr + row_idx)
    level_end = tl.load(level_ptr + row_idx + 1)

    # Process all rows in this level
    for level_row in range(level_start, level_end):
        row = tl.load(level_set + level_row)  # Get the actual row index from level set
        row_start = tl.load(Lp + row)
        row_end = tl.load(Lp + row + 1)

        # Load diagonal and RHS value
        diag_val = tl.load(Lx + row_end - 1)
        rhs = tl.load(x + row)

        # Process all non-diagonal elements in the row
        sum_val = 0.0
        for col in range(row_start, row_end - 1):
            col_idx = tl.load(Li + col)
            val = tl.load(Lx + col)
            x_val = tl.load(x + col_idx)
            sum_val += val * x_val

        # Store the result
        result = (rhs - sum_val) / diag_val
        tl.store(x + row, result)

def sptrsv_csr(Lp, Li, Lx, b, x, level_set, level_ptr, n, device):
    BLOCK_SIZE = 128  # Number of threads per block
    num_levels = len(level_ptr) - 1

    # Allocate Triton tensor for level scheduling
    level_ptr_triton = torch.tensor(level_ptr, dtype=torch.int32, device=device)
    level_set_triton = torch.tensor(level_set, dtype=torch.int32, device=device)

    # Launch the kernel
    sptrsv_csr_kernel[(num_levels,)](
        Lp, Li, Lx, x, level_set_triton, level_ptr_triton, n, BLOCK_SIZE=BLOCK_SIZE
    )

def preprocess_levels(Lp, Li, n):
    """ Precompute level scheduling information for sparse triangular solve """
    in_degree = [0] * n
    levels = [-1] * n
    level_ptr = [0]
    level_set = []

    # Calculate in-degree for each row
    for row in range(n):
        for i in range(Lp[row], Lp[row + 1]):
            in_degree[Li[i]] += 1

    # Compute levels using topological sorting
    queue = [i for i in range(n) if in_degree[i] == 0]
    while queue:
        level_ptr.append(len(level_set))
        next_queue = []
        for row in queue:
            level_set.append(row)
            for i in range(Lp[row], Lp[row + 1]):
                col = Li[i]
                in_degree[col] -= 1
                if in_degree[col] == 0:
                    next_queue.append(col)
        queue = next_queue

    level_ptr.append(len(level_set))
    return level_set, level_ptr

# Example: Testing the implementation
def test_sptrsv():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sparse matrix in CSR format
    Lp = torch.tensor([0, 2, 4, 7, 9], dtype=torch.int32, device=device)
    Li = torch.tensor([0, 1, 1, 2, 0, 2, 3, 2, 3], dtype=torch.int32, device=device)
    Lx = torch.tensor([4.0, 1.0, 2.0, 3.0, 1.0, 5.0, 2.0, 1.0, 3.0], dtype=torch.float32, device=device)

    # Right-hand side and solution vector
    b = torch.tensor([5.0, 5.0, 6.0, 4.0], dtype=torch.float32, device=device)
    x = b.clone()

    # Precomputed level scheduling information
    level_set, level_ptr = preprocess_levels(Lp.cpu().numpy(), Li.cpu().numpy(), len(b))

    # Solve
    sptrsv_csr(Lp, Li, Lx, b, x, level_set, level_ptr, len(b), device)

    # Verify
    print("Solution:", x.cpu().numpy())

# Run the test
if __name__ == "__main__":
    test_sptrsv()

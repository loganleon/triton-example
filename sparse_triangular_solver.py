import os
import torch
import triton
import triton.language as tl



# Triton kernel definition
@triton.jit
def sptrsv_csr_kernel(Lp, Li, Lx, b, x, n, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # Each block handles one row
    if row_idx >= n:
        return

    # Initialize the sum for this row
    sum_val = tl.zeros((1,), dtype=tl.float32)

    # Read the range of non-zero elements for this row
    row_start = tl.load(Lp + row_idx)
    row_end = tl.load(Lp + row_idx + 1)

    # Process all non-diagonal elements
    for idx in range(row_start, row_end - 1):
        col_idx = tl.load(Li + idx)
        val = tl.load(Lx + idx)
        sum_val += val * tl.load(x + col_idx)

    # Handle the diagonal element
    diag_idx = row_end - 1
    diag_val = tl.load(Lx + diag_idx)
    rhs = tl.load(b + row_idx)
    tl.store(x + row_idx, (rhs - sum_val) / diag_val)

def sptrsv_csr(Lp, Li, Lx, b, x, n):
    BLOCK_SIZE = 128
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # Ensure grid is a tuple
    sptrsv_csr_kernel[grid](Lp, Li, Lx, b, x, n, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    # Example sparse triangular matrix in CSR format
    # Matrix L:
    # [2   0   0   0]
    # [3   4   0   0]
    # [0   1   5   0]
    # [6   0   2   7]
    # RHS vector b: [4, 10, 15, 35]
    # Expected solution x: [2, 1, 2, 3]

    Lp = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int32).cuda()
    Li = torch.tensor([0, 0, 1, 1, 2, 0, 2, 3], dtype=torch.int32).cuda()
    Lx = torch.tensor([2, 3, 4, 1, 5, 6, 2, 7], dtype=torch.float32).cuda()
    b = torch.tensor([4, 10, 15, 35], dtype=torch.float32).cuda()
    x = torch.zeros(4, dtype=torch.float32).cuda()  # Solution vector

    # Solve the system
    n = 4
    sptrsv_csr(Lp, Li, Lx, b, x, n)

    # Print the solution
    print("Solution x:", x.cpu().numpy())
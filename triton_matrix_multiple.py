import time
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    # Program IDs for each block in the grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting indices of the block
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)

    # Create pointers to the blocks in A and B
    A_block_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize the accumulator
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_SIZE
    for k in range(0, K, BLOCK_SIZE):
        # Load blocks from A and B
        A_block = tl.load(A_block_ptr, mask=offs_m[:, None] < M)
        B_block = tl.load(B_block_ptr, mask=offs_n[None, :] < N)

        # Accumulate the product
        acc += tl.dot(A_block, B_block)

        # Update pointers for the next block
        A_block_ptr += BLOCK_SIZE * stride_ak
        B_block_ptr += BLOCK_SIZE * stride_bk

    # Write the result back to C
    C_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(A, B):
    # Get matrix dimensions
    M, K = A.shape
    Kb, N = B.shape

    # Output matrix
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define block size
    BLOCK_SIZE = 32

    # Define grid size
    grid = (
        (M + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    )

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return C

# Example usage
if __name__ == "__main__":
    # Initialize matrices
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)

    # Perform matrix multiplication using Triton
    start_time = time.time()
    C = triton_matmul(A, B)
    triton_time = time.time() - start_time
    print(f"Triton matmul time: {triton_time:.6f} seconds")

    # Verify correctness
    start_time = time.time()
    C_torch = torch.matmul(A, B)
    torch_time = time.time() - start_time
    print(f"PyTorch matmul time: {torch_time:.6f} seconds")

    max_diff = torch.max(torch.abs(C - C_torch))
    print(f"Max difference between Triton and PyTorch results: {max_diff}")

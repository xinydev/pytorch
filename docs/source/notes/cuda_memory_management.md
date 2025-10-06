# CUDA Memory Management in PyTorch

## Overview

This document provides an in-depth explanation of how PyTorch manages GPU memory allocation during model training and inference. Understanding these mechanisms can help you optimize memory usage and debug out-of-memory (OOM) errors.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [When Does Memory Allocation Happen?](#when-does-memory-allocation-happen)
- [Training Memory Allocation Timeline](#training-memory-allocation-timeline)
- [Inference Memory Allocation Timeline](#inference-memory-allocation-timeline)
- [CUDACachingAllocator Deep Dive](#cudacachingallocator-deep-dive)
- [Memory Optimization Best Practices](#memory-optimization-best-practices)
- [Debugging Memory Issues](#debugging-memory-issues)

## Architecture Overview

PyTorch uses a sophisticated **CUDACachingAllocator** (implemented in `c10/cuda/CUDACachingAllocator.cpp`) to manage GPU memory efficiently. This allocator operates at three levels:

```
┌─────────────────────────────────────────────────────────────┐
│ Python Level (torch.Tensor)                                  │
│ - Reference counting                                         │
│ - Automatic garbage collection                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ CUDACachingAllocator (C++)                                   │
│ - Memory pooling and caching                                 │
│ - Block management (split/merge)                             │
│ - Stream-aware allocation                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ CUDA Runtime API                                             │
│ - cudaMalloc / cudaFree                                      │
│ - Physical GPU memory management                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Memory Pooling**: Instead of calling `cudaMalloc`/`cudaFree` for every allocation, PyTorch maintains a cache pool
2. **Delayed Deallocation**: When a tensor is freed, memory is returned to the cache pool, not to CUDA
3. **Block Splitting and Merging**: Large blocks are split for small requests and merged when freed to reduce fragmentation
4. **Stream-Aware**: Each block tracks which CUDA streams use it to ensure safe reuse

## When Does Memory Allocation Happen?

GPU memory allocation occurs in the following scenarios:

### 1. **Tensor Creation**

Any operation that creates a new tensor on GPU triggers allocation:

```python
# Direct tensor creation
x = torch.randn(1000, 1000, device='cuda')  # ✓ Allocates ~4MB

# Moving tensor to GPU
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to('cuda')  # ✓ Allocates ~4MB

# In-place operations that change size
x = torch.empty(100, device='cuda')
x.resize_(1000, 1000)  # ✓ Allocates new memory

# Operations creating new tensors
y = x + 1  # ✓ Allocates memory for output
z = torch.matmul(x, y)  # ✓ Allocates memory for result
```

### 2. **Operations That DON'T Allocate**

Some operations reuse existing memory:

```python
x = torch.randn(1000, 1000, device='cuda')

# View operations (no allocation)
y = x.view(1000000)  # ✗ No allocation (shares memory with x)
z = x.t()  # ✗ No allocation (transposed view)
w = x[:500, :]  # ✗ No allocation (slice/view)

# In-place operations (no new allocation)
x.add_(1)  # ✗ Modifies x in place
x.mul_(2)  # ✗ In-place multiplication
```

### 3. **Gradient Allocation**

During backward pass, gradients are allocated:

```python
x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
y = x * 2

# Forward: allocates y
# Backward: allocates x.grad if not exists
y.backward(torch.ones_like(y))  # ✓ Allocates x.grad (~4MB)
```

## Training Memory Allocation Timeline

Here's what happens during a typical training iteration:

### Step-by-Step Breakdown

```python
# Example training loop
model = MyModel().cuda()  # ① Model parameters allocation
optimizer = torch.optim.Adam(model.parameters())  # ② Optimizer state allocation
criterion = nn.CrossEntropyLoss()

for batch_idx, (data, target) in enumerate(train_loader):
    # ③ Data transfer to GPU
    data = data.to('cuda')      # Allocates input batch
    target = target.to('cuda')  # Allocates target batch
    
    # ④ Forward pass
    optimizer.zero_grad()       # Resets gradients (no allocation)
    output = model(data)        # Allocates intermediate activations
    
    # ⑤ Loss computation
    loss = criterion(output, target)  # Allocates loss tensor
    
    # ⑥ Backward pass
    loss.backward()             # Allocates gradients for all parameters
    
    # ⑦ Optimizer step
    optimizer.step()            # May allocate temporary buffers (Adam: momentum, variance)
    
    # ⑧ Memory cleanup
    # Python GC frees references to data, output, loss
    # Memory returns to caching allocator pool
```

### Detailed Memory Allocation by Phase

#### Phase ① : Model Initialization

```python
model = nn.Linear(1024, 512).cuda()
```

**Allocations:**
- `model.weight`: `1024 × 512 × 4 bytes = 2 MB`
- `model.bias`: `512 × 4 bytes = 2 KB`

#### Phase ③ : Data Loading

```python
data = data.to('cuda')  # Shape: [batch_size, channels, height, width]
```

**Allocations:**
- Input batch: `batch_size × C × H × W × dtype_size`
- Example: `32 × 3 × 224 × 224 × 4 bytes ≈ 19 MB`

#### Phase ④ : Forward Pass

For each layer in the model:

```python
# Example: Conv2d -> BatchNorm2d -> ReLU
x = conv(x)      # ✓ Allocates output feature maps
x = bn(x)        # ✓ Allocates normalized output (may reuse conv output)
x = relu(x)      # ✗ In-place, no allocation (if inplace=True)
```

**Memory accumulation:**
- All intermediate activations are kept for backward pass
- Peak memory = sum of all layer outputs

#### Phase ⑥ : Backward Pass

```python
loss.backward()
```

**Allocations:**
- Gradient for each parameter: same size as parameter
- Temporary buffers for gradient computation
- Example for Linear layer:
  - `weight.grad`: same as `weight` (2 MB)
  - `bias.grad`: same as `bias` (2 KB)

#### Phase ⑦ : Optimizer Step

```python
optimizer.step()  # Adam optimizer
```

**Additional allocations (Adam):**
- Momentum buffer: same size as all parameters
- Variance buffer: same size as all parameters
- Total extra memory = 2 × parameter_memory

### Memory Timeline Visualization

```
Memory Usage Over One Training Iteration
│
│                    ┌─Peak Memory─┐
│                   ╱│             │╲
│                  ╱ │   Backward  │ ╲
│                 ╱  │   Gradients │  ╲
│                ╱   │             │   ╲
│          Forward   │             │    ╲
│     Activations    │             │     ╲
│    ╱          ╲    │             │      ╲
│   ╱            ╲   │             │       ╲
│  ╱              ╲  │             │        ╲
│ ╱Model+Data     ╲ │             │         ╲
│╱                 ╲│             │          ╲
├──────────────────────────────────────────────> Time
 ①②③  ④  Forward   ⑤⑥  Backward  ⑦  Step  ⑧
      Activations       Gradients
```

## Inference Memory Allocation Timeline

Inference has different memory characteristics than training:

### Key Differences

1. **No gradient storage**: `with torch.no_grad()` disables autograd
2. **No backward pass**: Only forward activations needed
3. **Memory can be freed immediately**: No need to keep intermediate results

### Inference Code Example

```python
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation
    for data in test_loader:
        # ① Data transfer
        data = data.to('cuda')  # Allocates input
        
        # ② Forward pass only
        output = model(data)  # Allocates output (smaller than training)
        
        # ③ Prediction
        pred = output.argmax(dim=1)  # Allocates small prediction tensor
        
        # ④ Memory freed automatically
        # No backward, so intermediate activations can be freed immediately
```

### Memory Allocation Breakdown

```python
# Training vs Inference Memory Comparison (approx.)
# For a typical ResNet-50 with batch_size=32:

# Training:
# - Model parameters: ~100 MB
# - Input batch: ~20 MB
# - Forward activations: ~300 MB (kept for backward)
# - Gradients: ~100 MB
# - Optimizer states (Adam): ~200 MB
# Total: ~720 MB

# Inference:
# - Model parameters: ~100 MB
# - Input batch: ~20 MB
# - Forward activations: ~50 MB (can be freed layer by layer)
# Total: ~170 MB (4x less than training)
```

## CUDACachingAllocator Deep Dive

### Block Structure

```cpp
struct Block {
    c10::DeviceIndex device;      // GPU device ID
    cudaStream_t stream;           // Allocation stream
    stream_set stream_uses;        // All streams using this block
    size_t size;                   // Block size in bytes
    size_t requested_size;         // Original requested size
    BlockPool* pool;               // Owning pool (small/large)
    void* ptr;                     // GPU memory address
    bool allocated;                // Currently in use?
    bool mapped;                   // Virtual memory mapped?
    Block* prev;                   // Previous block (for splitting)
    Block* next;                   // Next block (for splitting)
    int event_count;               // Outstanding CUDA events
    ExpandableSegment* expandable_segment_;  // For expandable segments
};
```

### Two-Pool Architecture

PyTorch separates allocations into two pools to optimize for different patterns:

```cpp
// Memory size constants
const size_t kMinBlockSize = 512;        // 512 bytes minimum
const size_t kSmallSize = 1048576;       // 1 MB threshold
const size_t kSmallBuffer = 2097152;     // 2 MB small pool buffer
const size_t kLargeBuffer = 20971520;    // 20 MB large pool buffer

// Two separate pools
class DeviceCachingAllocator {
    BlockPool small_blocks;  // For allocations <= 1 MB
    BlockPool large_blocks;  // For allocations > 1 MB
};
```

**Why two pools?**
- **Small Pool**: Packs many small tensors into 2 MB buffers, reducing fragmentation
- **Large Pool**: Large tensors get dedicated blocks to avoid waste

### Allocation Algorithm

The allocator follows this decision tree:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Round up size to 512-byte multiple                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Select pool: size <= 1MB ? small_pool : large_pool       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Search cache for free block >= requested_size            │
│    - Binary search in ordered set (by size, then stream)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │ Found?  │
                    └─┬─────┬─┘
                 YES  │     │ NO
                      ▼     ▼
            ┌─────────────────────────────────────────┐
            │ Split if too large                      │
            │ Return block                             │
            └─────────────────────────────────────────┘
                                  │
                                  ▼
                      ┌───────────────────────────────┐
                      │ 4. No cached block available  │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 5. Trigger memory callbacks   │
                      │    (retry cache search)       │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 6. Garbage collect old blocks │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 7. Call cudaMalloc()          │
                      │    - Release lock during call │
                      └──────────┬────────────────────┘
                                 │
                            ┌────┴────┐
                            │Success? │
                            └─┬─────┬─┘
                         YES  │     │ NO
                              ▼     ▼
                        Return  ┌─────────────────────┐
                        Block   │ 8. Release cached   │
                                │    blocks, retry    │
                                └──────────┬──────────┘
                                           │
                                      ┌────┴────┐
                                      │Success? │
                                      └─┬─────┬─┘
                                   YES  │     │ NO
                                        ▼     ▼
                                  Return  OutOfMemory
                                  Block   Exception
```

### Memory Release Flow

```python
# When a tensor goes out of scope:
x = torch.randn(1000, 1000, device='cuda')
del x  # or x goes out of scope
```

**What happens internally:**

```cpp
void free(Block* block) {
    block->allocated = false;
    
    // Check if block is used by other streams
    if (!block->stream_uses.empty()) {
        // Insert CUDA events to track completion
        for (stream : block->stream_uses) {
            event = create_event();
            cudaEventRecord(event, stream);
            cuda_events[stream].push_back({event, block});
        }
        return;  // Defer release until events complete
    }
    
    // Free immediately to cache pool
    free_block(block);
}

void free_block(Block* block) {
    // Try to merge with adjacent free blocks
    if (block->prev && !block->prev->allocated) {
        merge_blocks(block->prev, block);
    }
    if (block->next && !block->next->allocated) {
        merge_blocks(block, block->next);
    }
    
    // Insert back into free block pool
    block->pool->blocks.insert(block);
    
    // NOTE: cudaFree() is NOT called here!
    // Memory stays in cache for future reuse
}
```

### Stream-Aware Allocation

One of the most important features is stream awareness:

```python
# Example showing stream interaction
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    x = torch.randn(1000, 1000, device='cuda')  # Allocated on stream1
    
with torch.cuda.stream(stream2):
    y = x + 1  # Uses x from stream1
    
# Problem: x might be freed before stream2 finishes!
# Solution: record_stream()
x.record_stream(stream2)  # Tell allocator: stream2 also uses this memory
```

**How it works:**

```cpp
void recordStream(Block* block, CUDAStream stream) {
    if (stream == block->stream) {
        return;  // Same stream, no sync needed
    }
    
    // Add stream to the set of streams using this block
    block->stream_uses.insert(stream);
    
    // When freeing, insert CUDA event on this stream
    // Memory won't be reused until event completes
}
```

### Expandable Segments

A sophisticated feature to handle dynamic batch sizes:

**Problem:**
```python
# Training with varying batch sizes
for epoch in range(epochs):
    for batch in data_loader:
        if batch.size(0) == 32:  # Most batches
            output = model(batch)  # Uses 32-sized allocations
        elif batch.size(0) == 33:  # Occasional larger batch
            output = model(batch)  # Needs NEW allocations
                                   # → Old 32-sized blocks become fragmented
```

**Solution: Expandable Segments**

```cpp
// Instead of fixed-size cudaMalloc:
// 1. Reserve large virtual address space (256 TiB)
// 2. Map physical memory on demand (2 MB pages)
// 3. Expand segment when more memory needed

ExpandableSegment {
    CUdeviceptr ptr_;              // Virtual address start
    size_t mapped_size_;           // Currently mapped physical memory
    size_t max_handles_;           // Maximum virtual space
    vector<Handle> handles_;       // Physical memory handles
    
    SegmentRange map(size_t size) {
        // Allocate physical memory
        cuMemCreate(&handle, size, &prop, 0);
        
        // Map to virtual address
        cuMemMap(ptr_ + offset, size, 0, handle, 0);
        
        // Set access permissions
        cuMemSetAccess(ptr_ + offset, size, &desc, 1);
        
        return {ptr_ + offset, size};
    }
};
```

Enable with:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Memory Optimization Best Practices

### 1. Use Gradient Checkpointing

Trade computation for memory by recomputing activations during backward:

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
    
    def forward(self, x):
        # Without checkpointing: all activations kept
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        
        # With checkpointing: activations recomputed during backward
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint(self.layer3, x, use_reentrant=False)
        return x
```

**Memory savings:** ~50-70% for deep networks

### 2. Use Mixed Precision Training

FP16 uses half the memory of FP32:

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Memory savings:** ~40-50% for activations and gradients

### 3. Optimize Batch Size

Find the optimal batch size for your GPU:

```python
def find_optimal_batch_size(model, input_shape, max_batch_size=512):
    """Binary search to find largest batch size that fits in memory"""
    device = next(model.parameters()).device
    
    low, high = 1, max_batch_size
    optimal = 1
    
    while low <= high:
        batch_size = (low + high) // 2
        try:
            # Try this batch size
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            # Success! Try larger
            optimal = batch_size
            low = batch_size + 1
            
            # Clean up
            del dummy_input, output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Too large, try smaller
                high = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e
    
    return optimal
```

### 4. Use In-Place Operations

Reduce memory allocations with in-place ops:

```python
# Bad: allocates new tensors
x = x + 1
x = x * 2
x = torch.relu(x)

# Good: modifies in-place
x.add_(1)
x.mul_(2)
x = torch.relu(x, inplace=True)  # or F.relu(x, inplace=True)
```

### 5. Clear Cache When Needed

```python
# After memory-intensive operation
large_tensor = compute_something()
result = process(large_tensor)
del large_tensor

# Force release cached memory
torch.cuda.empty_cache()

# Note: This doesn't free memory used by live tensors!
# Only returns cached blocks to CUDA
```

### 6. Profile Memory Usage

```python
import torch

# Enable memory history tracking
torch.cuda.memory._record_memory_history(
    enabled=True,
    context='all',  # Record all contexts
    stacks='all',   # Record C++ and Python stacks
    max_entries=100000
)

# Run your code
model = MyModel().cuda()
for data in train_loader:
    output = model(data)
    loss = output.sum()
    loss.backward()
    break  # Just one iteration

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze at https://pytorch.org/memory_viz
```

## Debugging Memory Issues

### Common OOM Scenarios

#### 1. **OOM Despite Free Memory in nvidia-smi**

**Problem:** `nvidia-smi` shows free memory, but PyTorch OOMs

**Cause:** Memory fragmentation

**Solution:**
```python
# Option 1: Empty cache
torch.cuda.empty_cache()

# Option 2: Enable expandable segments
torch.cuda.memory._set_allocator_settings('expandable_segments:True')

# Option 3: Restart Python process (clears fragmentation)
```

#### 2. **Gradual Memory Growth**

**Problem:** Memory usage increases over iterations

**Cause:** Memory leaks from Python references

**Solution:**
```python
# Check for common leaks:
# 1. Accumulating lists
losses = []  # Bad!
for data in train_loader:
    loss = compute_loss(data)
    losses.append(loss)  # Keeps computation graph alive!

# Fix: detach and use item()
losses = []
for data in train_loader:
    loss = compute_loss(data)
    losses.append(loss.detach().cpu().item())  # Good!

# 2. Storing tensors in class attributes
class Model(nn.Module):
    def forward(self, x):
        self.last_output = x  # Bad! Creates reference
        return x

# Fix: Don't store or explicitly detach
class Model(nn.Module):
    def forward(self, x):
        self.last_output = x.detach()  # Good!
        return x
```

#### 3. **Multi-GPU OOM on One GPU**

**Problem:** Unbalanced load across GPUs

**Solution:**
```python
# Check memory per GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:")
    print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

# Use DistributedDataParallel instead of DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

### Memory Debugging Tools

```python
# 1. Get current memory stats
stats = torch.cuda.memory_stats()
print(f"Active allocations: {stats['active.all.current']}")
print(f"Reserved by PyTorch: {stats['reserved_bytes.all.current'] / 1e9:.2f} GB")

# 2. Memory summary
print(torch.cuda.memory_summary(device=0, abbreviated=False))

# 3. Get largest cached block
largest = torch.cuda.memory_stats()['inactive_split_bytes.all.peak']
print(f"Largest cached block: {largest / 1e9:.2f} GB")

# 4. Reset peak stats (for profiling specific code sections)
torch.cuda.reset_peak_memory_stats()
# ... run code ...
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")
```

## Configuration Options

Configure the allocator via environment variable:

```bash
# Syntax: PYTORCH_CUDA_ALLOC_CONF=option1:value1,option2:value2,...

# Example: Multiple options
export PYTORCH_CUDA_ALLOC_CONF=\
max_split_size_mb:512,\
garbage_collection_threshold:0.8,\
expandable_segments:True,\
roundup_power2_divisions:4
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_split_size_mb` | unlimited | Maximum size (MB) for block splitting |
| `roundup_power2_divisions` | 1 | Round allocation sizes to power-of-2 divisions |
| `garbage_collection_threshold` | 1.0 | Trigger GC when usage > threshold × total memory |
| `expandable_segments` | False | Enable expandable segment allocation |
| `backend` | native | Backend: `native` or `cudaMallocAsync` |

**Example use cases:**

```bash
# Reduce fragmentation for varying batch sizes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit memory for multi-process scenarios
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Aggressive garbage collection
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
```

## Advanced Topics

### CUDA Graphs and Memory

CUDA Graphs require special memory management:

```python
# Create a graph-private memory pool
g = torch.cuda.CUDAGraph()

# Warm-up (required!)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (memory addresses are fixed)
for data in test_loader:
    static_input.copy_(data)
    g.replay()
    # static_output contains result
```

**Key points:**
- Graph allocations use a private pool
- Memory addresses are frozen during replay
- Private pool persists until graph is destroyed

### Custom Memory Allocators

You can implement custom allocators:

```cpp
// my_allocator.cpp
extern "C" {
    void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
        void* ptr;
        cudaMalloc(&ptr, size);
        // Add custom logic here
        return ptr;
    }
    
    void my_free(void* ptr, size_t size, int device, cudaStream_t stream) {
        // Add custom logic here
        cudaFree(ptr);
    }
}
```

```python
# Use custom allocator
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator

allocator = CUDAPluggableAllocator('my_allocator.so', 'my_malloc', 'my_free')
change_current_allocator(allocator)
```

## References

- [CUDA Memory Management Documentation](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [Memory Visualizer Tool](https://pytorch.org/memory_viz)
- [CUDACachingAllocator Source Code](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)

## Summary

Key takeaways:

1. **PyTorch caches GPU memory** - Freed tensors return to pool, not CUDA
2. **Training uses 3-4x more memory than inference** - Due to activations and gradients
3. **Memory allocation happens at:**
   - Tensor creation
   - Forward pass (activations)
   - Backward pass (gradients)
   - Optimizer state updates
4. **Optimization techniques:**
   - Gradient checkpointing
   - Mixed precision training
   - Optimal batch sizing
   - In-place operations
5. **Debug with:**
   - `torch.cuda.memory_summary()`
   - Memory snapshots and visualizer
   - `nvidia-smi`

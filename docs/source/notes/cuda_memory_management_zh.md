# PyTorch CUDA 内存管理详解

## 概述

本文档深入解释PyTorch在模型训练和推理过程中如何管理GPU内存分配。理解这些机制可以帮助你优化内存使用并调试内存溢出（OOM）错误。

## 目录

- [架构概览](#架构概览)
- [什么时候会分配显存？](#什么时候会分配显存)
- [训练过程的内存分配时间线](#训练过程的内存分配时间线)
- [推理过程的内存分配时间线](#推理过程的内存分配时间线)
- [CUDACachingAllocator深度解析](#cudacachingallocator深度解析)
- [内存优化最佳实践](#内存优化最佳实践)
- [调试内存问题](#调试内存问题)

## 架构概览

PyTorch使用一个复杂的**CUDACachingAllocator**（CUDA缓存分配器，实现在`c10/cuda/CUDACachingAllocator.cpp`）来高效管理GPU内存。这个分配器在三个层次上运作：

```
┌─────────────────────────────────────────────────────────────┐
│ Python层 (torch.Tensor)                                      │
│ - 引用计数                                                    │
│ - 自动垃圾回收                                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ CUDACachingAllocator (C++)                                   │
│ - 内存池和缓存管理                                            │
│ - 块管理（分割/合并）                                         │
│ - 流感知分配                                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ CUDA Runtime API                                             │
│ - cudaMalloc / cudaFree                                      │
│ - 物理GPU内存管理                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **内存池化**：不为每次分配都调用`cudaMalloc`/`cudaFree`，而是维护一个缓存池
2. **延迟释放**：tensor释放时，内存返回缓存池而非返还CUDA
3. **块分割与合并**：大块分割给小请求使用，释放时合并以减少碎片
4. **流感知**：每个块追踪使用它的CUDA流，确保安全复用

## 什么时候会分配显存？

GPU内存分配发生在以下场景：

### 1. **Tensor创建**

任何在GPU上创建新tensor的操作都会触发分配：

```python
# 直接创建tensor
x = torch.randn(1000, 1000, device='cuda')  # ✓ 分配约4MB

# 将tensor移到GPU
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to('cuda')  # ✓ 分配约4MB

# 改变大小的原地操作
x = torch.empty(100, device='cuda')
x.resize_(1000, 1000)  # ✓ 分配新内存

# 创建新tensor的操作
y = x + 1  # ✓ 为输出分配内存
z = torch.matmul(x, y)  # ✓ 为结果分配内存
```

### 2. **不分配内存的操作**

某些操作复用现有内存：

```python
x = torch.randn(1000, 1000, device='cuda')

# 视图操作（不分配）
y = x.view(1000000)  # ✗ 不分配（与x共享内存）
z = x.t()  # ✗ 不分配（转置视图）
w = x[:500, :]  # ✗ 不分配（切片/视图）

# 原地操作（不分配新内存）
x.add_(1)  # ✗ 原地修改x
x.mul_(2)  # ✗ 原地乘法
```

### 3. **梯度分配**

反向传播期间会分配梯度：

```python
x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
y = x * 2

# 前向：分配y
# 反向：如果不存在则分配x.grad
y.backward(torch.ones_like(y))  # ✓ 分配x.grad（约4MB）
```

## 训练过程的内存分配时间线

下面是一个典型训练迭代中发生的事情：

### 逐步分解

```python
# 训练循环示例
model = MyModel().cuda()  # ① 模型参数分配
optimizer = torch.optim.Adam(model.parameters())  # ② 优化器状态分配
criterion = nn.CrossEntropyLoss()

for batch_idx, (data, target) in enumerate(train_loader):
    # ③ 数据传输到GPU
    data = data.to('cuda')      # 分配输入batch
    target = target.to('cuda')  # 分配目标batch
    
    # ④ 前向传播
    optimizer.zero_grad()       # 重置梯度（不分配）
    output = model(data)        # 分配中间激活值
    
    # ⑤ 损失计算
    loss = criterion(output, target)  # 分配损失tensor
    
    # ⑥ 反向传播
    loss.backward()             # 为所有参数分配梯度
    
    # ⑦ 优化器步骤
    optimizer.step()            # 可能分配临时缓冲区（Adam: 动量、方差）
    
    # ⑧ 内存清理
    # Python GC释放data、output、loss的引用
    # 内存返回缓存分配器池
```

### 各阶段详细内存分配

#### 阶段 ① : 模型初始化

```python
model = nn.Linear(1024, 512).cuda()
```

**分配内容：**
- `model.weight`: `1024 × 512 × 4字节 = 2 MB`
- `model.bias`: `512 × 4字节 = 2 KB`

#### 阶段 ③ : 数据加载

```python
data = data.to('cuda')  # 形状: [batch_size, channels, height, width]
```

**分配内容：**
- 输入batch: `batch_size × C × H × W × dtype大小`
- 示例: `32 × 3 × 224 × 224 × 4字节 ≈ 19 MB`

#### 阶段 ④ : 前向传播

对模型中的每一层：

```python
# 示例: Conv2d -> BatchNorm2d -> ReLU
x = conv(x)      # ✓ 分配输出特征图
x = bn(x)        # ✓ 分配归一化输出（可能复用conv输出）
x = relu(x)      # ✗ 原地操作，不分配（如果inplace=True）
```

**内存累积：**
- 所有中间激活值都保留用于反向传播
- 峰值内存 = 所有层输出的总和

#### 阶段 ⑥ : 反向传播

```python
loss.backward()
```

**分配内容：**
- 每个参数的梯度：与参数大小相同
- 梯度计算的临时缓冲区
- Linear层示例:
  - `weight.grad`: 与`weight`相同（2 MB）
  - `bias.grad`: 与`bias`相同（2 KB）

#### 阶段 ⑦ : 优化器步骤

```python
optimizer.step()  # Adam优化器
```

**额外分配（Adam）：**
- 动量缓冲区：与所有参数大小相同
- 方差缓冲区：与所有参数大小相同
- 总额外内存 = 2 × 参数内存

### 内存时间线可视化

```
一次训练迭代的内存使用
│
│                    ┌─峰值内存─┐
│                   ╱│         │╲
│                  ╱ │  反向   │ ╲
│                 ╱  │  梯度   │  ╲
│                ╱   │         │   ╲
│          前向     │         │    ╲
│       激活值      │         │     ╲
│    ╱        ╲    │         │      ╲
│   ╱          ╲   │         │       ╲
│  ╱            ╲  │         │        ╲
│ ╱模型+数据     ╲ │         │         ╲
│╱               ╲│         │          ╲
├──────────────────────────────────────────────> 时间
 ①②③  ④  前向     ⑤⑥  反向    ⑦  更新  ⑧
      激活值           梯度
```

## 推理过程的内存分配时间线

推理的内存特征与训练不同：

### 关键差异

1. **无梯度存储**：`with torch.no_grad()`禁用自动求导
2. **无反向传播**：只需要前向激活值
3. **内存可立即释放**：无需保留中间结果

### 推理代码示例

```python
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算
    for data in test_loader:
        # ① 数据传输
        data = data.to('cuda')  # 分配输入
        
        # ② 仅前向传播
        output = model(data)  # 分配输出（比训练小）
        
        # ③ 预测
        pred = output.argmax(dim=1)  # 分配小的预测tensor
        
        # ④ 内存自动释放
        # 无反向传播，中间激活值可立即释放
```

### 内存分配对比

```python
# 训练 vs 推理内存对比（近似值）
# 典型ResNet-50，batch_size=32:

# 训练:
# - 模型参数: ~100 MB
# - 输入batch: ~20 MB
# - 前向激活值: ~300 MB（为反向传播保留）
# - 梯度: ~100 MB
# - 优化器状态（Adam）: ~200 MB
# 总计: ~720 MB

# 推理:
# - 模型参数: ~100 MB
# - 输入batch: ~20 MB
# - 前向激活值: ~50 MB（可逐层释放）
# 总计: ~170 MB（比训练少4倍）
```

## CUDACachingAllocator深度解析

### Block结构

```cpp
struct Block {
    c10::DeviceIndex device;      // GPU设备ID
    cudaStream_t stream;           // 分配流
    stream_set stream_uses;        // 使用此块的所有流
    size_t size;                   // 块大小（字节）
    size_t requested_size;         // 原始请求大小
    BlockPool* pool;               // 所属池（small/large）
    void* ptr;                     // GPU内存地址
    bool allocated;                // 当前是否在使用？
    bool mapped;                   // 虚拟内存是否已映射？
    Block* prev;                   // 前一个块（用于分割）
    Block* next;                   // 下一个块（用于分割）
    int event_count;               // 未完成的CUDA事件
    ExpandableSegment* expandable_segment_;  // 用于可扩展段
};
```

### 双池架构

PyTorch将分配分为两个池以优化不同模式：

```cpp
// 内存大小常量
const size_t kMinBlockSize = 512;        // 512字节最小值
const size_t kSmallSize = 1048576;       // 1 MB阈值
const size_t kSmallBuffer = 2097152;     // 2 MB小池缓冲区
const size_t kLargeBuffer = 20971520;    // 20 MB大池缓冲区

// 两个独立的池
class DeviceCachingAllocator {
    BlockPool small_blocks;  // 用于分配 <= 1 MB
    BlockPool large_blocks;  // 用于分配 > 1 MB
};
```

**为什么要两个池？**
- **小池**：将许多小tensor打包到2 MB缓冲区中，减少碎片
- **大池**：大tensor获得专用块以避免浪费

### 分配算法

分配器遵循这个决策树：

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 将大小向上取整到512字节倍数                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 选择池: size <= 1MB ? small_pool : large_pool            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 在缓存中搜索 >= requested_size 的空闲块                   │
│    - 在有序集合中二分查找（按大小，然后按流）                │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │ 找到？  │
                    └─┬─────┬─┘
                 是  │     │ 否
                      ▼     ▼
            ┌─────────────────────────────────────────┐
            │ 如果太大则分割                           │
            │ 返回块                                   │
            └─────────────────────────────────────────┘
                                  │
                                  ▼
                      ┌───────────────────────────────┐
                      │ 4. 没有可用的缓存块            │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 5. 触发内存回调                │
                      │    （重试缓存搜索）            │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 6. 垃圾回收旧块                │
                      └──────────┬────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────────┐
                      │ 7. 调用cudaMalloc()            │
                      │    - 调用期间释放锁            │
                      └──────────┬────────────────────┘
                                 │
                            ┌────┴────┐
                            │ 成功？  │
                            └─┬─────┬─┘
                         是  │     │ 否
                              ▼     ▼
                        返回块  ┌─────────────────────┐
                                │ 8. 释放缓存块，重试  │
                                └──────────┬──────────┘
                                           │
                                      ┌────┴────┐
                                      │ 成功？  │
                                      └─┬─────┬─┘
                                   是  │     │ 否
                                        ▼     ▼
                                  返回块  内存溢出
                                          异常
```

### 内存释放流程

```python
# 当tensor超出作用域：
x = torch.randn(1000, 1000, device='cuda')
del x  # 或x超出作用域
```

**内部发生了什么：**

```cpp
void free(Block* block) {
    block->allocated = false;
    
    // 检查块是否被其他流使用
    if (!block->stream_uses.empty()) {
        // 插入CUDA事件以追踪完成
        for (stream : block->stream_uses) {
            event = create_event();
            cudaEventRecord(event, stream);
            cuda_events[stream].push_back({event, block});
        }
        return;  // 延迟释放直到事件完成
    }
    
    // 立即释放到缓存池
    free_block(block);
}

void free_block(Block* block) {
    // 尝试与相邻空闲块合并
    if (block->prev && !block->prev->allocated) {
        merge_blocks(block->prev, block);
    }
    if (block->next && !block->next->allocated) {
        merge_blocks(block, block->next);
    }
    
    // 插入回空闲块池
    block->pool->blocks.insert(block);
    
    // 注意：这里不调用cudaFree()！
    // 内存保留在缓存中供将来重用
}
```

### 流感知分配

最重要的特性之一是流感知：

```python
# 展示流交互的示例
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    x = torch.randn(1000, 1000, device='cuda')  # 在stream1上分配
    
with torch.cuda.stream(stream2):
    y = x + 1  # 使用来自stream1的x
    
# 问题：x可能在stream2完成前被释放！
# 解决方案：record_stream()
x.record_stream(stream2)  # 告诉分配器：stream2也使用这块内存
```

**工作原理：**

```cpp
void recordStream(Block* block, CUDAStream stream) {
    if (stream == block->stream) {
        return;  // 同一个流，不需要同步
    }
    
    // 将流添加到使用此块的流集合
    block->stream_uses.insert(stream);
    
    // 释放时，在此流上插入CUDA事件
    // 内存在事件完成前不会被重用
}
```

### 可扩展段（Expandable Segments）

处理动态batch size的高级特性：

**问题：**
```python
# 使用不同batch size训练
for epoch in range(epochs):
    for batch in data_loader:
        if batch.size(0) == 32:  # 大多数batch
            output = model(batch)  # 使用32大小的分配
        elif batch.size(0) == 33:  # 偶尔更大的batch
            output = model(batch)  # 需要新分配
                                   # → 旧的32大小块变成碎片
```

**解决方案：可扩展段**

```cpp
// 不使用固定大小的cudaMalloc：
// 1. 预留大量虚拟地址空间（256 TiB）
// 2. 按需映射物理内存（2 MB页）
// 3. 需要更多内存时扩展段

ExpandableSegment {
    CUdeviceptr ptr_;              // 虚拟地址起始
    size_t mapped_size_;           // 当前映射的物理内存
    size_t max_handles_;           // 最大虚拟空间
    vector<Handle> handles_;       // 物理内存句柄
    
    SegmentRange map(size_t size) {
        // 分配物理内存
        cuMemCreate(&handle, size, &prop, 0);
        
        // 映射到虚拟地址
        cuMemMap(ptr_ + offset, size, 0, handle, 0);
        
        // 设置访问权限
        cuMemSetAccess(ptr_ + offset, size, &desc, 1);
        
        return {ptr_ + offset, size};
    }
};
```

启用方式：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 内存优化最佳实践

### 1. 使用梯度检查点

用计算换内存，在反向传播时重新计算激活值：

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
    
    def forward(self, x):
        # 不使用检查点：保留所有激活值
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        
        # 使用检查点：反向传播时重新计算激活值
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint(self.layer3, x, use_reentrant=False)
        return x
```

**内存节省：**深层网络可节省约50-70%

### 2. 使用混合精度训练

FP16使用FP32一半的内存：

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    # 使用FP16前向传播
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 带梯度缩放的反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**内存节省：**激活值和梯度可节省约40-50%

### 3. 优化batch size

找到GPU的最优batch size：

```python
def find_optimal_batch_size(model, input_shape, max_batch_size=512):
    """二分查找能装入内存的最大batch size"""
    device = next(model.parameters()).device
    
    low, high = 1, max_batch_size
    optimal = 1
    
    while low <= high:
        batch_size = (low + high) // 2
        try:
            # 尝试这个batch size
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            # 成功！尝试更大的
            optimal = batch_size
            low = batch_size + 1
            
            # 清理
            del dummy_input, output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 太大了，尝试更小的
                high = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e
    
    return optimal
```

### 4. 使用原地操作

用原地操作减少内存分配：

```python
# 不好：分配新tensor
x = x + 1
x = x * 2
x = torch.relu(x)

# 好：原地修改
x.add_(1)
x.mul_(2)
x = torch.relu(x, inplace=True)  # 或 F.relu(x, inplace=True)
```

### 5. 必要时清理缓存

```python
# 在内存密集操作后
large_tensor = compute_something()
result = process(large_tensor)
del large_tensor

# 强制释放缓存内存
torch.cuda.empty_cache()

# 注意：这不会释放活跃tensor使用的内存！
# 只是将缓存块返回给CUDA
```

### 6. 分析内存使用

```python
import torch

# 启用内存历史追踪
torch.cuda.memory._record_memory_history(
    enabled=True,
    context='all',  # 记录所有上下文
    stacks='all',   # 记录C++和Python堆栈
    max_entries=100000
)

# 运行你的代码
model = MyModel().cuda()
for data in train_loader:
    output = model(data)
    loss = output.sum()
    loss.backward()
    break  # 只运行一次迭代

# 保存快照
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# 在 https://pytorch.org/memory_viz 分析
```

## 调试内存问题

### 常见OOM场景

#### 1. **尽管nvidia-smi显示有空闲内存但OOM**

**问题：**`nvidia-smi`显示有空闲内存，但PyTorch OOM

**原因：**内存碎片化

**解决方案：**
```python
# 选项1：清空缓存
torch.cuda.empty_cache()

# 选项2：启用可扩展段
torch.cuda.memory._set_allocator_settings('expandable_segments:True')

# 选项3：重启Python进程（清除碎片）
```

#### 2. **内存逐渐增长**

**问题：**内存使用随迭代增加

**原因：**Python引用导致的内存泄漏

**解决方案：**
```python
# 检查常见泄漏：
# 1. 累积列表
losses = []  # 不好！
for data in train_loader:
    loss = compute_loss(data)
    losses.append(loss)  # 保持计算图活跃！

# 修复：detach并使用item()
losses = []
for data in train_loader:
    loss = compute_loss(data)
    losses.append(loss.detach().cpu().item())  # 好！

# 2. 在类属性中存储tensor
class Model(nn.Module):
    def forward(self, x):
        self.last_output = x  # 不好！创建引用
        return x

# 修复：不存储或显式detach
class Model(nn.Module):
    def forward(self, x):
        self.last_output = x.detach()  # 好！
        return x
```

#### 3. **多GPU一个GPU OOM**

**问题：**GPU间负载不平衡

**解决方案：**
```python
# 检查每个GPU的内存
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:")
    print(f"  已分配: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"  已预留: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

# 使用DistributedDataParallel而不是DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

### 内存调试工具

```python
# 1. 获取当前内存统计
stats = torch.cuda.memory_stats()
print(f"活跃分配: {stats['active.all.current']}")
print(f"PyTorch预留: {stats['reserved_bytes.all.current'] / 1e9:.2f} GB")

# 2. 内存摘要
print(torch.cuda.memory_summary(device=0, abbreviated=False))

# 3. 获取最大缓存块
largest = torch.cuda.memory_stats()['inactive_split_bytes.all.peak']
print(f"最大缓存块: {largest / 1e9:.2f} GB")

# 4. 重置峰值统计（用于分析特定代码段）
torch.cuda.reset_peak_memory_stats()
# ... 运行代码 ...
peak = torch.cuda.max_memory_allocated()
print(f"峰值内存: {peak / 1e9:.2f} GB")
```

## 配置选项

通过环境变量配置分配器：

```bash
# 语法: PYTORCH_CUDA_ALLOC_CONF=选项1:值1,选项2:值2,...

# 示例：多个选项
export PYTORCH_CUDA_ALLOC_CONF=\
max_split_size_mb:512,\
garbage_collection_threshold:0.8,\
expandable_segments:True,\
roundup_power2_divisions:4
```

### 可用选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `max_split_size_mb` | 无限制 | 块分割的最大大小（MB） |
| `roundup_power2_divisions` | 1 | 将分配大小向上取整到2的幂的除数 |
| `garbage_collection_threshold` | 1.0 | 当使用量 > 阈值 × 总内存时触发GC |
| `expandable_segments` | False | 启用可扩展段分配 |
| `backend` | native | 后端：`native`或`cudaMallocAsync` |

**使用示例：**

```bash
# 减少不同batch size的碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 限制多进程场景的内存
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 积极垃圾回收
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
```

## 高级主题

### CUDA Graphs和内存

CUDA Graphs需要特殊的内存管理：

```python
# 创建graph私有内存池
g = torch.cuda.CUDAGraph()

# 预热（必需！）
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# 捕获
with torch.cuda.graph(g):
    static_output = model(static_input)

# 重放（内存地址固定）
for data in test_loader:
    static_input.copy_(data)
    g.replay()
    # static_output包含结果
```

**要点：**
- Graph分配使用私有池
- 重放时内存地址冻结
- 私有池持续存在直到graph销毁

## 总结

关键要点：

1. **PyTorch缓存GPU内存** - 释放的tensor返回池，而非CUDA
2. **训练使用3-4倍推理的内存** - 由于激活值和梯度
3. **内存分配发生在：**
   - Tensor创建
   - 前向传播（激活值）
   - 反向传播（梯度）
   - 优化器状态更新
4. **优化技术：**
   - 梯度检查点
   - 混合精度训练
   - 最优batch size
   - 原地操作
5. **调试工具：**
   - `torch.cuda.memory_summary()`
   - 内存快照和可视化工具
   - `nvidia-smi`

## 参考资料

- [CUDA内存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [内存可视化工具](https://pytorch.org/memory_viz)
- [CUDACachingAllocator源码](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)

.. _cpp-language-extensions:

5.4. C/C++ 语言扩展
===================

.. _function-and-variable-annotations:

5.4.1. 函数和变量注释
---------------------

.. _execution-space-specifiers:

5.4.1.1. 执行空间说明符
^^^^^^^^^^^^^^^^^^^^^^^

执行空间说明符 ``__host__``、``__device__`` 和 ``__global__`` 指示函数是在主机还是设备上执行。

.. list-table:: 执行空间说明符
   :header-rows: 1
   :widths: 25 15 15 15 15

   * - 执行空间说明符
     - 执行位置
     - 
     - 可调用位置
     - 
   * - 
     - 主机
     - 设备
     - 主机
     - 设备
   * - ``__host__``，无说明符
     - ✅
     - ❌
     - ✅
     - ❌
   * - ``__device__``
     - ❌
     - ✅
     - ❌
     - ✅
   * - ``__global__``
     - ❌
     - ✅
     - ✅
     - ✅
   * - ``__host__ __device__``
     - ✅
     - ✅
     - ✅
     - ✅

``__global__`` 函数的限制：

- 必须返回 ``void``。
- 不能是 ``class``、``struct`` 或 ``union`` 的成员。
- 需要执行配置，如 `内核配置 <#execution-configuration>`__ 中所述。
- 不支持递归。
- 有关其他限制，请参阅 ``__global__`` `函数参数 <cpp-language-support.html#global-function-parameters>`__。

对 ``__global__`` 函数的调用是异步的。它们在设备完成执行之前返回到主机线程。

用 ``__host__ __device__`` 声明的函数同时为主机和设备编译。可以使用 ``__CUDA_ARCH__`` `宏 <#cuda-arch-macro>`__ 区分主机和设备代码路径：

.. code-block:: c++

   __host__ __device__ void func() {
   #if defined(__CUDA_ARCH__)
       // 设备代码路径
   #else
       // 主机代码路径
   #endif
   }

.. _memory-space-specifiers:

5.4.1.2. 内存空间说明符
^^^^^^^^^^^^^^^^^^^^^^^

内存空间说明符 ``__device__``、``__managed__``、``__constant__`` 和 ``__shared__`` 指示设备上变量的存储位置。

.. list-table:: 内存空间说明符
   :header-rows: 1
   :widths: 20 25 25 20

   * - 内存空间说明符
     - 位置
     - 可访问者
     - 生存期
   * - ``__device__``
     - 设备全局内存
     - 设备线程（grid）/ CUDA Runtime API
     - 程序/CUDA 上下文
   * - ``__constant__``
     - 设备常量内存
     - 设备线程（grid）/ CUDA Runtime API
     - 程序/CUDA 上下文
   * - ``__managed__``
     - 主机和设备（自动）
     - 主机/设备线程
     - 程序
   * - ``__shared__``
     - 设备（流多处理器）
     - 块线程
     - 块
   * - 无说明符
     - 设备（寄存器）
     - 单个线程
     - 单个线程

- ``__device__`` 和 ``__constant__`` 变量都可以通过 `CUDA Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`__ 函数 ``cudaGetSymbolAddress()``、``cudaGetSymbolSize()``、``cudaMemcpyToSymbol()`` 和 ``cudaMemcpyFromSymbol()`` 从主机访问。

- ``__constant__`` 变量在设备代码中是只读的，只能使用 `CUDA Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`__ 从主机修改。

以下示例说明如何使用这些 API：

.. code-block:: c++

   __device__   float device_var       = 4.0f; // 设备内存中的变量
   __constant__ float constant_mem_var = 4.0f; // 常量内存中的变量
                                               // 为便于阅读，以下示例侧重于设备变量。
   int main() {
       float* device_ptr;
       cudaGetSymbolAddress((void**) &device_ptr, device_var);        // 获取 device_var 的地址

       size_t symbol_size;
       cudaGetSymbolSize(&symbol_size, device_var);                   // 检索符号的大小（4 字节）。

       float host_var;
       cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // 从设备复制到主机。

       host_var = 3.0f;
       cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // 从主机复制到设备。
   }

.. _shared-memory:

5.4.1.2.1. ``__shared__`` 内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__shared__`` 内存变量可以具有静态大小（在编译时确定）或动态大小（在内核启动时确定）。有关在运行时指定共享内存大小的详细信息，请参阅 `内核配置 <#execution-configuration>`__ 部分。

共享内存限制：

- 具有动态大小的变量必须声明为外部数组或指针。
- 具有静态大小的变量不能在其声明中初始化。

以下示例说明如何声明和确定 ``__shared__`` 变量的大小：

.. code-block:: c++

   extern __shared__ char dynamic_smem_pointer[];
   // extern __shared__ char* dynamic_smem_pointer; 替代语法

   __global__ void kernel() { // 或 __device__ 函数
       __shared__ int smem_var1[4];                  // 静态大小
       auto smem_var2 = (int*) dynamic_smem_pointer; // 动态大小
   }

   int main() {
       size_t shared_memory_size = 16;
       kernel<<<1, 1, shared_memory_size>>>();
       cudaDeviceSynchronize();
   }

.. _managed-memory:

5.4.1.2.2. ``__managed__`` 内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__managed__`` 变量具有以下限制：

- ``__managed__`` 变量的地址不是常量表达式。
- ``__managed__`` 变量不应具有引用类型 ``T&``。
- 当 CUDA 运行时可能未处于有效状态时，不应使用 ``__managed__`` 变量的地址或值，包括以下情况：
  
  - 在具有 ``static`` 或 ``thread_local`` 存储期的对象的静态/动态初始化或销毁中。
  - 在调用 ``exit()`` 后执行的代码中。例如，标记为 ``__attribute__((destructor))`` 的函数。
  - 在 CUDA 运行时可能未初始化时执行的代码中。例如，标记为 ``__attribute__((constructor))`` 的函数。

- ``__managed__`` 变量不能用作 ``decltype()`` 表达式的未括号化 id 表达式参数。
- ``__managed__`` 变量具有与`动态分配的托管内存 <../02-basics/understanding-memory.html#memory-unified-memory>`__ 指定的相同的一致性和一致性行为。
- 另请参阅 `局部变量 <cpp-language-support.html#local-variables>`__ 的限制。

.. _inline-specifiers:

5.4.1.3. 内联说明符
^^^^^^^^^^^^^^^^^^^

以下说明符可用于控制 ``__host__`` 和 ``__device__`` 函数的内联：

- ``__noinline__``：指示 ``nvcc`` 不要内联该函数。
- ``__forceinline__``：强制 ``nvcc`` 在单个翻译单元内内联该函数。
- ``__inline_hint__``：使用 `链接时优化 <../02-basics/nvcc.html#nvcc-link-time-optimization>`__ 时启用跨翻译单元的积极内联。

这些说明符是互斥的。

.. _restrict-pointers:

5.4.1.4. ``__restrict__`` 指针
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 通过 ``__restrict__`` 关键字支持受限指针。

当两个或多个指针引用重叠的内存区域时，就会发生指针别名。这可能会抑制代码重排序和公共子表达式消除等优化。

受限限定指针是程序员的一个承诺，即在该指针的生存期内，它指向的内存只能通过该指针访问。这允许编译器执行更积极的优化。

- 访问设备函数的所有线程只从中读取；或者
- 最多有一个线程写入其中，没有其他线程从中读取。

以下示例说明别名问题，并演示如何使用受限指针帮助编译器减少指令数量：

.. code-block:: c++

   __device__
   void device_function(const float* a, const float* b, float* c) {
       c[0] = a[0] * b[0];
       c[1] = a[0] * b[0];
       c[2] = a[0] * b[0] * a[1];
       c[3] = a[0] * a[1];
       c[4] = a[0] * b[0];
       c[5] = b[0];
       ...
   }

由于指针 ``a``、``b`` 和 ``c`` 可能别名化，通过 ``c`` 的任何写入都可能修改 ``a`` 或 ``b`` 的元素。为保证功能正确性，编译器无法将 ``a[0]`` 和 ``b[0]`` 加载到寄存器中、相乘并将结果存储在 ``c[0]`` 和 ``c[1]`` 中。这是因为如果 ``a[0]`` 和 ``c[0]`` 位于同一位置，结果将与抽象执行模型不同。编译器无法利用公共子表达式。类似地，编译器无法重排 ``c[4]`` 的计算与 ``c[0]`` 和 ``c[1]`` 的计算，因为前面的 ``c[3]`` 写入可能会改变 ``c[4]`` 计算的输入。

通过将 ``a``、``b`` 和 ``c`` 声明为受限指针，程序员通知编译器这些指针没有别名。这意味着写入 ``c`` 永远不会覆盖 ``a`` 或 ``b`` 的元素。这会将函数原型更改如下：

.. code-block:: c++

   __device__
   void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);

请注意，所有指针参数都必须是受限的，编译器优化器才能有效。添加 ``__restrict__`` 关键字后，编译器可以随意重排并执行公共子表达式消除，同时保持与抽象执行模型相同的功能。

.. code-block:: c++

   __device__
   void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
       float t0 = a[0];
       float t1 = b[0];
       float t2 = t0 * t1;
       float t3 = a[1];
       c[0]     = t2;
       c[1]     = t2;
       c[4]     = t2;
       c[2]     = t2 * t3;
       c[3]     = t0 * t3;
       c[5]     = t1;
       ...
   }

结果是减少了内存访问和计算次数，但由于在寄存器中缓存加载和公共子表达式，寄存器压力增加。

由于寄存器压力是许多 CUDA 代码中的关键问题，使用受限指针可能会通过降低占用率对性能产生负面影响。

访问标记为 ``__restrict__`` 的 ``__global__`` 函数 ``const`` 指针被编译为只读缓存加载，类似于 `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`__ ``ld.global.nc`` 或 ``__ldg()`` `低级加载和存储函数 <#low-level-load-store-functions>`__ 指令。

.. code-block:: c++

   __global__
   void kernel1(const float* in, float* out) {
       *out = *in; // PTX: ld.global
   }

   __global__
   void kernel2(const float* __restrict__ in, float* out) {
       *out = *in;  // PTX: ld.global.nc
   }

.. _grid-constant-parameters:

5.4.1.5. ``__grid_constant__`` 参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``__grid_constant__`` 注释 ``__global__`` 函数参数可防止编译器创建该参数的每线程副本。相反，grid 中的所有线程将通过单个地址访问该参数，这可以提高性能。

``__grid_constant__`` 参数具有以下属性：

- 它具有内核的生存期。
- 它对于单个内核是私有的，意味着该对象不可被来自其他 grid 的线程访问，包括子 grid。
- 内核中的所有线程看到相同的地址。
- 它是只读的。修改 ``__grid_constant__`` 对象或其任何子对象（包括 ``mutable`` 成员）是未定义行为。

要求：

- 用 ``__grid_constant__`` 注释的内核参数必须具有 ``const`` 限定的非引用类型。
- 所有函数声明必须与任何 ``__grid_constant__`` 参数一致。
- 函数模板特化必须与主模板声明匹配，关于任何 ``__grid_constant__`` 参数。
- 函数模板实例化也必须与主模板声明匹配，关于任何 ``__grid_constant__`` 参数。

示例：

.. code-block:: c++

   struct MyStruct {
       int         x;
       mutable int y;
   };

   __device__ void external_function(const MyStruct&);

   __global__ void kernel(const __grid_constant__ MyStruct s) {
       // s.x++; // 编译错误：尝试修改只读内存
       // s.y++; // 未定义行为：尝试修改只读内存

       // 编译器不会创建 "s" 的每线程本地副本：
       external_function(s);
   }

.. _annotation-summary:

5.4.1.6. 注释摘要
^^^^^^^^^^^^^^^^^

.. list-table:: 注释摘要
   :header-rows: 1
   :widths: 25 25 25 25

   * - 注释
     - ``__host__`` / ``__device__`` / ``__host__ __device__``
     - ``__global__``
     - 
   * - `__noinline__ <#inline-specifiers>`__、`__forceinline__ <#inline-specifiers>`__、`__inline_hint__ <#inline-specifiers>`__
     - 函数
     - ❌
   * - `__restrict__ <#restrict>`__
     - 指针参数
     - 指针参数
   * - `__grid_constant__ <#grid-constant>`__
     - ❌
     - 参数
   * - `__launch_bounds__ <#launch-bounds>`__
     - ❌
     - 函数
   * - `__maxnreg__ <#maximum-number-of-registers-per-thread>`__
     - ❌
     - 函数
   * - `__cluster_dims__ <#cluster-dimensions>`__
     - ❌
     - 函数

.. _built-in-types-and-variables:

5.4.2. 内建类型和变量
---------------------

.. _host-compiler-type-extensions:

5.4.2.1. 主机编译器类型扩展
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 允许使用非标准算术类型，只要主机编译器支持。支持以下类型：

- 128 位整数类型 ``__int128``。
  
  - 当主机编译器定义 ``__SIZEOF_INT128__`` 宏时，在 Linux 上支持。

- 128 位浮点类型 ``__float128`` 和 ``_Float128`` 在计算能力 10.0 及更高版本的 GPU 设备上可用。``__float128`` 类型的常量表达式可能由编译器以较低精度的浮点表示形式处理。
  
  - 当主机编译器定义 ``__SIZEOF_FLOAT128__`` 或 ``__FLOAT128__`` 宏时，在 Linux x86 上支持。

- ``_Complex`` `类型 <https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html>`__ 仅在主机代码中支持。

.. _built-in-variables:

5.4.2.2. 内建变量
^^^^^^^^^^^^^^^^^

用于指定和检索沿 x、y 和 z 维度的 grid 和块的内核配置的值是 ``dim3`` 类型。用于获取块和线程索引的变量是 ``uint3`` 类型。``dim3`` 和 ``uint3`` 都是由三个名为 ``x``、``y`` 和 ``z`` 的无符号值组成的简单结构。在 C++11 及更高版本中，``dim3`` 的所有组件的默认值为 1。

仅限设备的内建变量：

- ``dim3 gridDim``：包含 grid 的维度，即沿 x、y 和 z 维度的线程块数量。

- ``dim3 blockDim``：包含线程块的维度，即沿 x、y 和 z 维度的线程数量。

- ``uint3 blockIdx``：包含 grid 内的块索引，沿 x、y 和 z 维度。

- ``uint3 threadIdx``：包含块内的线程索引，沿 x、y 和 z 维度。

- ``int warpSize``：运行时值，定义为 warp 中的线程数，通常为 ``32``。另请参阅 `Warp 和 SIMT <../01-introduction/programming-model.html#programming-model-warps-simt>`__ 了解 warp 的定义。

.. _built-in-types:

5.4.2.3. 内建类型
^^^^^^^^^^^^^^^^^

CUDA 提供从基本整数和浮点类型派生的向量类型，这些类型在主机和设备上都受支持。

.. list-table:: 向量类型
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - C++ 基本类型
     - 向量 X1
     - 向量 X2
     - 向量 X3
     - 向量 X4
   * - ``signed char``
     - ``char1``
     - ``char2``
     - ``char3``
     - ``char4``
   * - ``unsigned char``
     - ``uchar1``
     - ``uchar2``
     - ``uchar3``
     - ``uchar4``
   * - ``signed short``
     - ``short1``
     - ``short2``
     - ``short3``
     - ``short4``
   * - ``unsigned short``
     - ``ushort1``
     - ``ushort2``
     - ``ushort3``
     - ``ushort4``
   * - ``signed int``
     - ``int1``
     - ``int2``
     - ``int3``
     - ``int4``
   * - ``unsigned``
     - ``uint1``
     - ``uint2``
     - ``uint3``
     - ``uint4``
   * - ``signed long``
     - ``long1``
     - ``long2``
     - ``long3``
     - ``long4_16a/long4_32a``
   * - ``unsigned long``
     - ``ulong1``
     - ``ulong2``
     - ``ulong3``
     - ``ulong4_16a/ulong4_32a``
   * - ``signed long long``
     - ``longlong1``
     - ``longlong2``
     - ``longlong3``
     - ``longlong4_16a/longlong4_32a``
   * - ``unsigned long long``
     - ``ulonglong1``
     - ``ulonglong2``
     - ``ulonglong3``
     - ``ulonglong4_16a/ulonglong4_32a``
   * - ``float``
     - ``float1``
     - ``float2``
     - ``float3``
     - ``float4``
   * - ``double``
     - ``double1``
     - ``double2``
     - ``double3``
     - ``double4_16a/double4_32a``

向量类型是结构体。它们的第一个、第二个、第三个和第四个组件分别可以通过 ``x``、``y``、``z`` 和 ``w`` 字段访问。

.. code-block:: c++

   int sum(int4 value) {
       return value.x + value.y + value.z + value.w;
   }

它们都有一个 ``make_<type_name>()`` 形式的工厂函数；例如：

.. code-block:: c++

   int4 add_one(int x, int y, int z, int w) {
       return make_int4(x + 1, y + 1, z + 1, w + 1);
   }

如果主机代码不是用 ``nvcc`` 编译的，可以通过包含 CUDA toolkit 中提供的 ``cuda_runtime.h`` 头文件导入向量类型和相关函数。

.. _kernel-configuration:

5.4.3. 内核配置
---------------

对 ``__global__`` 函数的任何调用都必须为该调用指定*执行配置*。此执行配置定义将用于在设备上执行函数的 grid 和块的维度，以及关联的 `流 <../02-basics/asynchronous-execution.html#cuda-streams>`__。

执行配置通过在函数名和括号参数列表之间插入 ``<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>`` 形式的表达式来指定，其中：

- ``grid_dim`` 是 `dim3 <#built-in-variables>`__ 类型，指定 grid 的维度和大小，使得 ``grid_dim.x * grid_dim.y * grid_dim.z`` 等于正在启动的块数；

- ``block_dim`` 是 `dim3 <#built-in-variables>`__ 类型，指定每个块的维度和大小，使得 ``block_dim.x * block_dim.y * block_dim.z`` 等于每个块的线程数；

- ``dynamic_smem_bytes`` 是可选的 ``size_t`` 参数，默认为零。它指定此次调用每个块动态分配的共享内存字节数，除了静态分配的内存。此内存由 ``extern __shared__`` 数组使用（参见 `__shared__ 内存 <#shared-memory-specifier>`__）。

- ``stream`` 是 ``cudaStream_t``（指针）类型，指定关联的流。``stream`` 是可选参数，默认为 ``NULL``。

以下示例显示内核函数声明和调用：

.. code-block:: c++

   __global__ void kernel(float* parameter);

   kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);

执行配置的参数在实际函数参数之前计算。

如果 ``grid_dim`` 或 ``block_dim`` 超过设备允许的最大大小（如 `计算能力 <compute-capabilities.html#compute-capabilities>`__ 中指定），或者 ``dynamic_smem_bytes`` 大于考虑静态分配内存后的可用共享内存，函数调用将失败。

.. _thread-block-cluster:

5.4.3.1. 线程块集群
^^^^^^^^^^^^^^^^^^^

计算能力 9.0 及更高版本允许用户指定编译时线程块集群维度，以便内核可以使用 CUDA 中的 `集群层次结构 <../02-basics/intro-to-cuda-cpp.html#thread-block-clusters>`__。可以使用 ``__cluster_dims__`` 属性指定编译时集群维度，语法如下：``__cluster_dims__([x, [y, [z]]])``。以下示例显示 X 维度为 2、Y 和 Z 维度为 1 的编译时集群大小。

.. code-block:: c++

   __global__ void __cluster_dims__(2, 1, 1) kernel(float* parameter);

``__cluster_dims__()`` 的默认形式指定内核将作为 grid 集群启动。如果未指定集群维度，用户可以在启动时指定。如果在启动时未能指定维度，将导致启动时错误。

线程块集群的维度也可以在运行时指定，可以使用 ``cudaLaunchKernelEx`` API 启动带有集群的内核。此 API 接受 ``cudaLaunchConfig_t`` 类型的配置参数、内核函数指针和内核参数。

.. _launch-bounds:

5.4.3.2. 启动边界
^^^^^^^^^^^^^^^^^

如 `内核启动和占用率 <../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy>`__ 部分所述，使用更少的寄存器允许更多线程和线程块驻留在多处理器上，从而提高性能。

因此，编译器使用启发式方法最小化寄存器使用，同时将 `寄存器溢出 <../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-registers>`__ 和指令计数保持在最低限度。应用程序可以通过使用 ``__launch_bounds__()`` 限定符在 ``__global__`` 函数定义中指定启动边界，以编译器提供额外信息的形式可选地辅助这些启发式方法：

.. code-block:: c++

   __global__ void
   __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
   MyKernel(...) {
       ...
   }

- ``maxThreadsPerBlock`` 指定应用程序将启动 ``MyKernel()`` 的每个块的最大线程数；它编译为 ``.maxntid`` PTX 指令。

- ``minBlocksPerMultiprocessor`` 是可选的，指定每个多处理器所需的最小驻留块数；它编译为 ``.minnctapersm`` PTX 指令。

- ``maxBlocksPerCluster`` 是可选的，指定应用程序将启动 ``MyKernel()`` 的每个集群的最大线程块数；它编译为 ``.maxclusterrank`` PTX 指令。

如果指定了启动边界，编译器首先推导内核应使用的寄存器数量的上限 ``L``。这确保 ``minBlocksPerMultiprocessor`` 个 ``maxThreadsPerBlock`` 线程块可以驻留在多处理器上（如果未指定 ``minBlocksPerMultiprocessor``，则为单个块）。然后编译器优化寄存器使用：

- 如果初始寄存器使用超过 ``L``，编译器会减少它，直到它小于或等于 ``L``。这通常会导致本地内存使用增加和/或指令数量增加。

- 如果初始寄存器使用低于 ``L``
  
  - 如果指定了 ``maxThreadsPerBlock`` 但未指定 ``minBlocksPerMultiprocessor``，编译器使用 ``maxThreadsPerBlock`` 确定从 ``n`` 到 ``n + 1`` 个驻留块过渡的寄存器使用阈值。
  
  - 如果同时指定了 ``minBlocksPerMultiprocessor`` 和 ``maxThreadsPerBlock``，编译器可能会将寄存器使用增加到 ``L``，以减少指令数量并更好地隐藏单线程指令的延迟。

如果使用以下方式执行，内核将无法启动：

- 每块线程数超过其启动边界 ``maxThreadsPerBlock``。
- 每集群线程块数超过其启动边界 ``maxBlocksPerCluster``。

.. _maximum-number-of-registers-per-thread:

5.4.3.3. 每线程最大寄存器数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了启用低级性能调优，CUDA C++ 提供 ``__maxnreg__()`` 函数限定符，它将性能调优信息传递给后端优化编译器。``__maxnreg__()`` 限定符指定可以分配给线程块中单个线程的最大寄存器数。在 ``__global__`` 函数定义中：

.. code-block:: c++

   __global__ void
   __maxnreg__(maxNumberRegistersPerThread)
   MyKernel(...) {
       ...
   }

``maxNumberRegistersPerThread`` 变量指定要分配给内核 ``MyKernel()`` 的线程块中单个线程的最大寄存器数；它编译为 ``.maxnreg`` PTX 指令。

``__launch_bounds__()`` 和 ``__maxnreg__()`` 限定符不能一起应用于同一内核。

可以使用 ``--maxrregcount <N>`` 编译器选项控制文件中所有 ``__global__`` 函数的寄存器使用。对于具有 ``__maxnreg__`` 限定符的内核函数，此选项将被忽略。

.. _synchronization-primitives:

5.4.4. 同步原语
---------------

.. _thread-block-synchronization-functions:

5.4.4.1. 线程块同步函数
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __syncthreads();
   int  __syncthreads_count(int predicate);
   int  __syncthreads_and(int predicate);
   int  __syncthreads_or(int predicate);

这些内建函数协调同一块内线程之间的通信。当块中的线程访问共享或全局内存中的相同地址时，可能会发生读后写、写后读或写后写冲突。可以通过在这些访问之间同步线程来避免这些冲突。

这些内建函数具有以下语义：

- ``__syncthreads*()`` 等待线程块中所有未退出的线程同时到达程序中相同的 ``__syncthreads*()`` 内建调用或退出。

- ``__syncthreads*()`` 在参与线程之间提供内存排序：对 ``__syncthreads*()`` 内建的调用强烈发生在（参见 `C++ 规范 [intro.races] <https://eel.is/c++draft/intro.races>`__）任何参与线程从等待中解除阻塞或退出之前。

以下示例显示如何使用 ``__syncthreads()`` 同步线程块内的线程并安全地对线程之间共享的数组元素求和：

.. code-block:: c++

   // 假设 blockDim.x 为 128
   __global__ void example_syncthreads(int* input_data, int* output_data) {
       __shared__ int shared_data[128];
       // 每个线程写入 'shared_data' 的不同元素：
       shared_data[threadIdx.x] = input_data[threadIdx.x];

       // 所有线程同步，保证所有对 'shared_data' 的写入在
       // 任何线程从 '__syncthreads()' 解除阻塞之前排序：
       __syncthreads();

       // 单个线程安全读取 'shared_data'：
       if (threadIdx.x == 0) {
           int sum = 0;
           for (int i = 0; i < blockDim.x; ++i) {
               sum += shared_data[i];
           }
           output_data[blockIdx.x] = sum;
       }
   }

``__syncthreads*()`` 内建函数允许在条件代码中使用，但仅当条件在整个线程块中统一求值时。否则，执行可能会挂起或产生意外的副作用。

**带谓词的 ``__syncthreads()`` 变体：**

``int __syncthreads_count(int predicate);`` 与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词，并返回谓词计算为非零值的线程数。

``int __syncthreads_and(int predicate);`` 与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词。当且仅当谓词对所有线程都计算为非零值时，它返回非零值。

``int __syncthreads_or(int predicate);`` 与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词。当且仅当谓词对一个或多个线程计算为非零值时，它返回非零值。

.. _warp-synchronization-function:

5.4.4.2. Warp 同步函数
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __syncwarp(unsigned mask = 0xFFFFFFFF);

内建函数 ``__syncwarp()`` 协调同一 warp 内线程之间的通信。当 warp 内的一些线程访问共享或全局内存中的相同地址时，可能会发生潜在的读后写、写后读或写后写冲突。可以通过在这些访问之间同步线程来避免这些数据冲突。

调用 ``__syncwarp(mask)`` 在 ``mask`` 中命名的 warp 内参与线程之间提供内存排序：对 ``__syncwarp(mask)`` 的调用强烈发生在（参见 `C++ 规范 [intro.races] <https://eel.is/c++draft/intro.races>`__）``mask`` 中命名的任何 warp 线程从等待中解除阻塞或退出之前。

这些函数受 `Warp __sync 内建约束 <#warp-sync-intrinsic-constraints>`__ 限制。

.. _memory-fence-functions:

5.4.4.3. 内存栅栏函数
^^^^^^^^^^^^^^^^^^^^^

CUDA 编程模型采用弱排序内存模型。换句话说，CUDA 线程将数据写入共享内存、全局内存、页锁定主机内存或对等设备内存的顺序不一定是另一个 CUDA 或主机线程观察数据写入的顺序。在没有内存栅栏或同步的情况下从同一内存位置读取或写入会导致未定义行为。

内存栅栏和同步函数强制执行内存访问的 `顺序一致性排序 <https://en.cppreference.com/w/cpp/atomic/memory_order>`__。这些函数在强制执行排序的 `线程作用域 <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes>`__ 方面有所不同，但独立于访问的内存空间，包括共享内存、全局内存、页锁定主机内存和对等设备的内存。

.. hint::

   建议尽可能使用 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html>`__ 提供的 ``cuda::atomic_thread_fence`` 以确保安全和可移植性。

**块级内存栅栏**

.. code-block:: c++

   // <cuda/atomic> 头文件
   cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);

确保：

- 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有写入，被调用线程所在块中的所有线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入之前；

- 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有读取，排序在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有读取之前。

**设备级内存栅栏**

.. code-block:: c++

   cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

确保：

- 调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入，不会被设备中的任何线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的任何写入之前。

**系统级内存栅栏**

.. code-block:: c++

   cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);

确保：

- 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有写入，被设备中的所有线程、主机线程和对等设备中的所有线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入之前。

.. note::

   有关 C/C++ 语言扩展的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html>`_。
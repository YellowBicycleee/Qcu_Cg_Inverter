# gauge shift

## 以t轴为例

我们的维度的长度分别为 $Lx$, $Ly$, $Lz$, $Lt$，但是在shift的时候只交换边界部分，每个维度设置两个buffer，长度均为 $Lx \times Ly \times Lz \times 1 \times Nd \times Nc \times Nc$，因为是t的边界，所以t维度只有一行！

由于gauge是全部能够拿到的，所以不区分奇偶，一起做shift。
# MPI Dslash

## 1. 使用到的各种类型，以及模块

## 2. 默认满足的条件，以及约定俗成的规定

$grid_x, grid_y, grid_z, grid_t$ 均大于等于1

## 3. 处理边界点的办法

拿t轴为例，
- 当该轴块数是1的时候，t轴不分块，边缘部分直接计算；
- 当该轴块数大于1的时候，t轴分块，计算到t轴部分后，前边界收缩1格, 后边界收缩1格，因为边界部分计算需要相邻格子的数据，等中心部分计算完毕后，再去进行边缘部分的计算
### 3.1 y z t轴

### 3.2 x轴

#### 前向边界
上述方案遇到一个问题：y拿不到，eo算不出来，
- $parity == 0$ 时，
	- 如果 $eo_{new} = (z+t) \mod 2 == 0$ ,那么y算成奇数坐标，这样 $eo == 1 $，then $y = 2y_{sub}+1$，前向边界点坐标为 $(Lx_{sub}-1, 2y_{sub}+1, z, t)$
	- 如果 $eo_{new} = (z+t) \mod 2 == 1$，那么y算成偶数坐标，这样 $eo == 1 $，then $y = 2y_{sub}$ 前向边界点坐标为 $(Lx_{sub}-1, 2y_{sub}+0, z, t)$
	- 综上前边界坐标是 $(Lx_{sub}-1, 2y_{sub}+(parity==eo_{new}), z, t)$
- $parity == 1$ 的时候，
	- 如果 $eo_{new} == (z+t) \mod 2 == 0$，那么y算成偶数坐标，这样 $eo == 0$，then $y = 2y_{sub}$，前向边界点为 $(Lx_{sub}-1, 2y_{sub}+0, z, t)$
	- 如果 $eo_{new} == (z+t) \mod 2 == 1$，那么y算成奇数坐标，这样 $eo == 0$，then $y = 2y_{sub}+1$，前向边界点为 $(Lx_{sub}-1, 2y_{sub}+1, z, t)$
	- 综上前边界坐标是 $(Lx_{sub}-1, 2y_{sub}+(parity==eo_{new}), z, t)$

#### 后向边界
- $parity == 0$ 时，
	- 如果 $eo_{new} = (z+t) \mod 2 == 0$ ,那么y算成偶数坐标，这样 $eo == 0$，then $y = 2y_{sub}$，后向边界点坐标为 $(0, 2y_{sub}+0, z, t)$
	- 如果 $eo_{new} = (z+t) \mod 2 == 1$，那么y算成奇数坐标，这样 $eo == 0$，then $y = 2y_{sub}+1$，后向边界点坐标为 $(0, 2y_{sub}+1, z, t)$
	- 综上前边界坐标是 $(0, 2y_{sub}+(parity \neq eo_{new}), z, t)$
- $parity == 1$ 的时候，
	- 如果 $eo_{new} == (z+t) \mod 2 == 0$，那么y算成奇数坐标，这样 $eo == 1$，then $y = 2y_{sub}+1$，后向边界点为 $(0, 2y_{sub}+1, z, t)$
	- 如果 $eo_{new} == (z+t) \mod 2 == 1$，那么y算成偶数坐标，这样 $eo == 1$，then $y = 2y_{sub}+0$，后向边界点为 $(Lx_{sub}-1, 2y_{sub}+0, z, t)$
	- 综上后边界坐标是 $(0, 2y_{sub}+(parity \neq eo_{new}), z, t)$

- 综上，在MPI程序中，收缩 $Ly_{sub} = \frac{Ly}{2}$，另外 $eo_{new} = (z+t) \mod 2$
	- 前边界坐标是 $(Lx_{sub}-1, 2y_{sub}+(parity==eo_{new}), z, t)$
	- 后边界坐标是 $(0, 2y_{sub}+(parity \neq eo_{new}), z, t)$


### 接收方，奇偶性反过来，使用1-parity

### 总之
- 计算点的边界坐标
	- 综上前边界坐标是 $(Lx_{sub}-1, 2y_{sub}+(parity==eo_{new}), z, t)$
	- 后边界坐标是 $(0, 2y_{sub}+(parity \neq eo_{new}), z, t)$

- 如果认为parity是计算点的奇偶性，那么1-parity是src的奇偶性
	- 前边界坐标是 $(Lx_{sub}-1, 2y_{sub}+(1-parity==eo_{new}), z, t)$
	- 后边界坐标是 $(0, 2y_{sub}+(1-parity \neq eo_{new}), z, t)$
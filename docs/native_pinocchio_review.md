# 原生 Pinocchio 接口审查与修复

这次使用真正的 Pinocchio Python / C++ 绑定重新运行测试，
没有用 NumPy 测试替身代替原生计算，也没有连接或使能机械臂。

目前确认的问题主要位于项目对原生 API 的封装和模型使用。
RS 完整位姿误差的 Jlog6 修正已通过原生数值求导检查；
不能据这些封装错误认定 Pinocchio 核心算法存在缺陷。

## 1. 发现的问题与改动

| 位置 | 原行为与影响 | 修复 |
| --- | --- | --- |
| `kinematics/robot_model.py` | 用 `joint ID` 索引 `lower/upperPositionLimit`，限位错位，最后一个关节越界 | 使用该关节的 `idx_q`；拒绝把多维配置关节当作标量限位 |
| `dynamics/robot_model.py` | `pin.Motion(g)` 传入 3 维向量，原生构造器报错；`g.linear.x` 也不是 NumPy API | 使用 `pin.Motion(g, zeros(3))`；通过数组读取并返回副本，校验有限 3 维重力 |
| `dynamics/inertia.py` | `computeAllTerms` 不更新 `data.C`，但封装直接返回它，得到零值或历史值 | 额外调用 `computeCoriolisMatrix`，保持返回 `(M,C,g)` 的原契约 |
| 科氏矩阵及动力学导数 | 原生遍历只更新树结构中的相关块，复用或改动 Data 会残留跨分支元素 | 计算前清零相关输出矩阵；对外返回独立副本 |
| `dynamics/derivatives.py` | 调用不存在的 `computeMassMatrixDerivatives` 和 `data.dMassdq` | 利用 RNEA 解析导数与单位加速度，逐列计算质量矩阵的配置导数 |
| 同上 | 读取不存在的 `data.dtau_da` | 复制 `computeRNEADerivatives` 返回的三元组；第三项对应 `data.M` |
| 同上 | 把配置导数维度写成 `nq` | 按配置切空间 `nv` 求导，适用于 `nq != nv` 的浮动基座等模型 |
| `dynamics/centroidal.py` | 调用不存在的 `computeCentroidalVelocities` | 使用 `centerOfMass(model,data,q,v)`，读取 `data.vcom[0]` |
| `kinematics/inverse_kinematics.py` | 仅位置模式使用随末端旋转的 LOCAL 误差，却忽略误差坐标系随 q 变化的导数 | 位置误差改为世界系，雅可比线速度部分旋转到同一坐标系；完整 SE(3) 模式保持 LOCAL + Jlog6 |
| 配置补齐与限位 | 全零补齐会产生非法四元数；逐分量裁剪破坏多维关节配置 | 使用 `pin.neutral` 补齐，校验归一化；仅裁剪一维位置关节，随机重试使用原生 `randomConfiguration` |
| 帧与输入校验 | 不存在的帧会返回 `model.nframes`，后续数组访问越界；超长 q 被静默截断 | 提前拒绝不存在的帧、非法维度和非有限 q；受控配置前缀不得截断多维关节 |
| 模型加载 | 全局缓存会保留旧硬件 YAML 或被其他调用者修改的重力 | 增加显式 `hardware_config_path`，取消共享可变模型及旧配置缓存；重力控制器使用实际机器人配置 |
| `dynamics/forward_dynamics.py` | 奇异模型的原生 ABA 可能返回 NaN | 检查质量矩阵正定，拒绝奇异模型；该封装增加 CRBA / Cholesky 成本，不作为纯 O(n) 实时接口 |
| `tests/conftest.py` | 缺少原生库时自动使用替身，容易把运动学测试通过误认为原生验证 | 默认必须使用原生库；仅显式 `--offline-kinematics` 开启替身，原生测试标记为跳过 |

质量矩阵导数采用恒等式：在速度为零时，
`rnea(q,0,e_k) = M(q)e_k + g(q)`。
将这次 RNEA 的配置解析导数减去重力导数，就得到 M 第 k 列的配置导数。
返回张量为 `(nv,nv,nv)`，第一维是求导方向，后两维是质量矩阵行列。
它需要 `nv+1` 次解析导数调用，适合分析、校验和优化，未放入伺服循环。

原生 Python 的 CRBA / computeAllTerms 绑定已经补齐质量矩阵的对称部分，
本次验证没有发现“Python 质量矩阵只有上三角”的问题。

## 2. 模型检查

RS 原生模型包含 6 个机械臂转动自由度及 2 个夹爪移动自由度，`nq=nv=8`。
硬件机械臂控制只求解前 6 轴，两个夹爪模型自由度保持 neutral。
夹爪电机角度不能直接写入 URDF 的移动关节，若要考虑开合位置对模型的影响，
还需要实际夹爪传动的角度到位移映射。

RS 当前 URDF 在测试构型下质量矩阵正定，重力与零速度、零加速度的 RNEA 一致。
模型中没有已辨识的电机静摩擦、Stribeck 摩擦、减速器滞回、驱动器内部速度估计，
原生刚体动力学的正确性不能代替这些实机参数的标定。

同时检查到 DM URDF 的两个手指自由度惯量为零。
完整模型质量矩阵奇异，原生 ABA 输出 NaN；6 轴机械臂子模型可以计算。
需要分析固定夹爪的机械臂时，可使用：

```python
locked = [i for i in range(1, model.njoints) if model.joints[i].idx_q >= 6]
arm_model = pin.buildReducedModel(model, locked, pin.neutral(model))
```

如果要模拟手指运动，应填入测量或可信 CAD 的质量和惯量。
本次没有编造这些参数，也没有改动 URDF。

## 3. 对低速上下抖动的意义

这次原生验证支持之前的修复方向：完整位姿的误差雅可比与数值导数一致，
10 cm / 8 s 的 RS 参考轨迹可以连续求解并保持 z 单调。
恢复真正的重力计算后，还覆盖了 250 Hz、8 秒、2001 次控制回调，
确认命令有限、参考速度正常结束、全程没有软件故障。
测试中的电机 I/O 是替身，不代表真实机械臂跟踪能力。

`computeAllTerms` 的旧 C 和几个缺失 API 是动力学公共接口的问题；
脚本 8 的重力前馈调用 `computeGeneralizedGravity`，不经过这些函数。
因此这些错误不能直接解释该脚本的上下抖动。
更直接的代码问题仍是之前修复的离散目标保持、零目标速度、固定 IK 死区、
零空间限位项泄漏及反馈/控制时序。

## 4. 验证与复现

在独立临时 Windows / Python 3.11 环境验证，用户原有 Anaconda 环境没有被修改。
测试覆盖 RS / DM 模型、配置索引、重力、M/C/g/nle、RNEA/ABA、质心和能量恒等式、
切空间导数、帧错误及 Data 复用，并覆盖此前的轨迹与故障回归。

| 后端 | 结果 |
| --- | --- |
| 原生 Pinocchio 3.9.0 / Python 3.11.16 | 46 项全部通过 |
| 原生 Pinocchio 4.1.0 / Python 3.11.16 | 46 项全部通过 |
| 显式 NumPy 替身 / Python 3.12.14 | 22 项通过，24 项原生验证跳过 |

第三行只验证替身模式的行为，不是项目支持 Python 3.12 的声明。
项目声明的运行版本仍为 Python 3.10 / 3.11。

Windows 原生验证环境可以使用 conda-forge 创建：

```powershell
conda create -n rebotarm-native -c conda-forge python=3.11 pinocchio=3.9.0 numpy pytest pyyaml pip
conda activate rebotarm-native
python -m pip install motorbridge==0.5.0
python -m pytest tests -v
```

项目的 Python 3.11 锁文件使用 Pinocchio 4.1.0，该版本也已完成相同验证，
无需为此次修复切换或强制降级依赖版本。
正常运行测试时缺少原生库会明确失败。
只需要有限运动学替身回归时，明确运行：

```powershell
python -m pytest tests --offline-kinematics -v
```

该模式会跳过所有原生接口与动力学测试。两种模式都不连接电机。

## 5. 原生源码依据

- [官方 3.9 逆运动学实现：Jlog6 与 integrate](https://github.com/stack-of-tasks/pinocchio/blob/v3.9.0/examples/inverse-kinematics.py)
- [computeAllTerms Python 绑定及实际更新项](https://github.com/stack-of-tasks/pinocchio/blob/v3.9.0/bindings/python/algorithm/expose-cat.cpp)
- [RNEA 导数 Python 绑定：返回值及 data.M](https://github.com/stack-of-tasks/pinocchio/blob/v3.9.0/bindings/python/algorithm/expose-rnea-derivatives.cpp)
- [Motion 原生构造器与数组分量](https://github.com/stack-of-tasks/pinocchio/blob/v3.9.0/include/pinocchio/bindings/python/spatial/motion.hpp)
- [Pinocchio 官方安装说明](https://github.com/stack-of-tasks/pinocchio#installation)

# DM 低速抖动：源码修复与真机验证

适用本项目的 DM 4340P / 4310 机械臂，电脑通过达妙串口桥连接电机侧 CAN。
Windows、Ubuntu、macOS 分别运行，一个进程打开一次适配器。
**保留原来的 500 Hz 控制和指令发送频率。** 这次没有连接真机；软件回归不能证明机械摩擦已经消除。

## 改动和原因

| 层次 | 发现的问题 | 修改 |
| --- | --- | --- |
| 脚本 / 配置 | COM 和 `/dev/cu` 被送到 CAN 控制器构造函数 | 显式 `transport: dm-serial`，支持 COM、tty、cu；脚本增加 `--channel`、`--mode` |
| DM 模型 | 不能套用 RS 的关节方向和模型；DM URDF 有两个被动夹爪自由度 | 使用 DM 自己的 URDF、六轴关节顺序和限位；夹爪电机角度不填到两个手指关节里 |
| 轨迹 / IK | 与 RS 共用的低速小步丢失、参考阶梯、速度不连续问题 | 复用已经修复的精确 IK、连续 C2 路径与解析 q/qd/qdd；两个 DM 模式新增完整 10 cm / 8 s 回归 |
| 控制模式 | POS_VEL 实际只发位置和速度上限，却计算了未发送的力矩 | POS_VEL 日志中的力矩前馈为零，启用摩擦力矩时要求 MIT；MIT 实际发送 q、qd、重力及可选摩擦前馈 |
| 原生反馈 | Rust 缓存有接收时间，Python ABI 丢弃时间；旧缓存每次读取都刷新本地时间 | 新增明确的 C ABI，原子快照包含位置、实际接收年龄、新报文序号和 SDK 量程；只有新的传感器报文更新序号 |
| 反馈线程 | 断线或某轴无反馈时，旧数据可能继续被当作新数据 | 六轴全部收到新的传感器反馈才发布，失败保留旧时间；100 ms 失鲜故障保持；收到故障状态在下一次控制回调锁存故障 |
| 串口收发 | 同一 mutex 覆盖串口读取等待和写入 | 对已经打开的串口使用 `try_clone`，分开 RX/TX 锁；整个发送包仍在 TX 锁内写完，禁止交错 |
| 发送开销 | 六轴分别经过 Python/native 调用和串口写入 | 同组六轴一次 native 调用、一次 `write_all`，串联六条完整的 30 字节报文；夹爪保持独立模式和发送 |
| 多余报文 | 高频控制已经带来反馈，反馈线程仍重复查询 | 优先使用尚未发布的新传感器报文；只向缺少新报文的轴查询，静止预检仍可主动查询 |
| 协议量程 | SDK 的 PMAX/VMAX/TMAX 与电机固件不一致会同时扭曲解码和 MIT 编码 | 启动前只读 21/22/23 寄存器，与实际 SDK 模型量程比对；不一致拒绝启动，不自动改固件量程 |
| 时序诊断 | 配置为 500 Hz，并不代表实际达到 500 Hz | 状态记录目标/实际平均回调频率、最长回调耗时、超过两个周期的次数，以及 DM 原生反馈接收速率估计；分析器检验实际频率 |

原生补丁由 `tools/build_motorbridge_feedback.py` 下载固定、校验过 SHA256 的 motorbridge 0.5.6 源码后生成。
`tools/patches/dm_native_patch.py` 和两个 Rust 模板保留全部修改，便于审查与在其他平台重新构建。
没有猜测 Rust 内部内存布局，也没有额外打开一个串口或 PCAN。

源码依据：

- [0.5.6 DM 反馈、协议和量程](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_vendors/damiao/src/motor.rs)
- [0.5.6 串口封装与共用锁](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_core/src/dm_serial.rs)
- [0.5.6 Python ABI 状态字段](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_abi/src/state_ffi.rs)

## 保留 500 Hz 的处理

当前配置保持 `rate: 500`，串口 `baud: 921600`，`serial_link: auto`。
`auto` 表示尚未核实板子的实际 USB/UART 链路，**不是自动证明带宽足够**。
官方板子的具体型号目前未提供，不能仅凭 COM / tty / cu 或“官方板子”判断传输上限。

若确实经过真实 UART，8N1 下 7 × 30 × 10 × 500 = 1,050,000 bit/s，
还未计入额外查询就超过 921600。此时批量写能减少调用开销，却不能减少线上字节数。
确认是真实 UART 后设 `serial_link: uart`：启动按控制频率和最坏情况下反馈查询检查预算；
如果桥接器和固件均支持更高波特率，可以保持 500 Hz，并配置双方确认支持的波特率。
例如 2 Mbaud 在计算上有余量，**这里没有替硬件确认支持，也没有自动调整**。

若确认是直接 USB CDC，设 `serial_link: usb-cdc`。CDC 的 line coding 波特率可能是名义参数，
不能把上面的 UART 算式当作实际 USB 吞吐上限。要检查完整控制下的实际周期、反馈年龄、
反馈接收速率，以及 CAN 侧是否积压。未知板型仍保留 auto，并按相同方式验收。

本轮使用收发隔离、同组批量写、减少重复反馈请求来保留频率。
只读预检和控制启动会用同样的批量格式查询六轴，必须收齐所有轴的新回复；
如果特定旧桥固件不能正确解析连续报文，会在使能前失败。可设置 `motion_control.dm_batch_send: false`
恢复逐条写入，频率仍为 500 Hz，再用日志验收；不要跳过失败强行运行。
批量写仍发送全部六轴每周期的新目标，没有跳过微小位移、合并成广播 CAN 指令，
也没有把电机指令频率偷偷改成 250 Hz。改变桥接报文格式需要板子固件配合，本轮保持原格式。

如果 USB/桥固件/CAN 侧实际容量不足，另一个可行路径是 DM 电机改接已有 PCAN：
配置 `transport: can` 和正确的 channel。电机侧仍是 CAN，必须匹配电机的实际 CAN 波特率。
经典 CAN 中一次命令往往还伴随一次反馈；7 轴 × 500 Hz 的双向报文在 1 Mbps 下也可能接近上限，
不能认为换 PCAN 就一定足够，需实测总线负载。不要未经电机固件确认直接切 CAN FD。

## 安装与只读预检

在实际控制使用的 Python 3.10/3.11 原生 Pinocchio 环境中执行：

```text
python -m pip install motorbridge==0.5.6 pytest
python tools/build_motorbridge_feedback.py
python -m pytest tests -q
```

需要 Rust 和对应平台 C 链接器。重启 Python 后本项目自动选择 `.motorbridge/<平台>/` 下的库；
显式 `MOTORBRIDGE_LIB` 优先。只升级 wheel 不包含本地补丁。
同一份构建同时支持 RS、直接 CAN 和 DM 串口桥；不包括另一路 `DM_Device` 专用 USB SDK。
Windows 库已在本工作区构建。Ubuntu、macOS 的原生构建和驱动仍须在相应系统验收。

三个平台端口示例，实际名称以本机枚举为准：

```text
python tools/dm_hardware_check.py --channel COM7 --seconds 10
python tools/dm_hardware_check.py --channel /dev/ttyACM0 --seconds 10
python tools/dm_hardware_check.py --channel /dev/cu.usbmodem101 --seconds 10
```

任选对应平台的一条。工具只读位置、量程和模型重力，不改模式、不使能、不清零、退出不失能。
查看 `logs/dm_preflight.json`：补丁、所有关节量程匹配、限位内、无错误、采样跨度 ≤20 ms。
`ready_for_control` 表示上述低频反馈预检通过，**没有证明完整 500 Hz 真机控制已经通过**。
URDF 限位内也不能替代零点、方向和装配检查；DM joint2/3 的有效区间与 RS 相反。

## 从静止到 10 cm / 8 s

机械臂有支撑、零点方向正确、运动空间足够，急停可操作。运动脚本退出会失能。
先保持默认 `friction.enabled: false`，默认 POS_VEL 可用以下命令进入：

```text
python example/8_arm_traj_control.py --hw rebotarm_dm.yaml --channel COM7 --mode posvel
```

Ubuntu/macOS 替换 channel。也可先测试 `--mode mit`；更换模式需要结束当前控制、重新启动，
不能在一段运动中直接切换。如果原来的 POS_VEL 保持已经稳定，先记录该模式再比较 MIT。

交互命令：

```text
status
end_state
record
```

等待 10 s，再输入 `log logs/dm_hold.csv`，查看状态中的实际平均回调频率、最长回调耗时、
反馈年龄和原生反馈接收速率。500 Hz 的名义周期是 2 ms；如果长期只有 250 Hz、
回调耗时接近/超过 2 ms 或反馈陈旧，应先处理传输和主机调度，再讨论摩擦。
50 Hz 反馈线程只发布快照；原生报文序号仍能估算实际接收速率。不同固件的反馈策略可能不同，
这个速率不是电机内部控制频率，也不是严格的一发一收计数。

静止稳定后分别测试关节 2、3 的小幅正反向运动：

```text
joint joint2 0.02 2
```

等 `moving: false, goal_reached: true, fault: None`，保存 `log logs/dm_joint2.csv`。
然后做相反方向，保存另一份日志。确认关节姿态允许，关节轨迹不包含碰撞规划。
接着按 1 cm / 2 s、5 cm / 4 s、10 cm / 8 s 分级测试，每段先到位、保存，再运行下一段。

```text
lift 0.01 2
lift 0.05 4
lift 0.1 8
```

这些不是要求一次连发三条。每段都记录，例如最后一段 `log logs/dm_lift_8s.csv`。
确认返回路径可行后，返回适当起点，分别比较 10 cm / 4 s、8 s、16 s，
并在相同起点、负载、模式和增益下比较。不要无限累加向上位移。

分析命令：

```text
python tools/analyze_motion_log.py logs/dm_hold.csv
python tools/analyze_motion_log.py logs/dm_joint2.csv --joint-only
python tools/analyze_motion_log.py logs/dm_lift_8s.csv --expected-dz 0.1 --output logs/dm_lift_report.json
```

日志包含实际硬件配置的时序阈值；DM 反馈年龄按 100 ms 验收。
有实际循环统计时，运行满 1 s 后平均频率低于目标的 90% 会判定失败。
同时检查全部控制周期的 P99；500 Hz 下超过 4 ms 会判定周期波动检查失败。
这是初步诊断阈值，全周期统计与日志采样的区别见
[进一步优化](control_timing_optimization.md)。
该指标只能证实主机回调频率，不能代替电机收到指令的时刻。
默认到位误差 ≤2 mm、反向最大回退 ≤0.5 mm、静止 Z 峰峰值 ≤1 mm，
这些是初次诊断阈值，不是对机器精度的承诺。

## 还有抖动时如何定位

- 参考 Z 也上下变化：看路径/IK校验、起点漂移、限位或奇异位置；本轮测试验证了 DM 参考路径单调。
- 参考平滑、主机周期超时或反馈年龄异常：看串口写等待、板子缓存、CAN 负载、USB 驱动和调度；不能先归因于摩擦。
- 时序正常、静止就抖：检查零点/方向、重力模型、负载、增益、机构间隙及松动。
- 慢速正反向才出现滞留后跳动：再做单关节摩擦辨识，逐轴、有限幅度测试；MIT 可启用已实现的平滑 Stribeck 前馈。
- POS_VEL 中的爬行：调节电机内部位置/速度环与积分，或通过 MIT 对比重力和摩擦前馈；POS_VEL 的第二个字段是速度上限，不能当轨迹速度前馈。

默认模型下 MIT 的位置量化步长约为 `25 / 65535 = 0.0003815 rad`；
4340P 的速度步长约为 `20 / 4095 = 0.004884 rad/s`，4310 约为 `60 / 4095 = 0.01465 rad/s`。
这会限制非常低速时的速度前馈分辨率。POS_VEL 用浮点位置/速度上限，但内部摩擦、编码器分辨率
和 PID 仍存在。不能只改 PMAX/VMAX 缩小范围而不同时匹配固件、编码和力矩限幅。

DM URDF 两个无质量手指使完整模型 ABA 的质量矩阵奇异；重力/RNEA 可用于当前六轴控制，
需要正向动力学时应显式约束被动关节并构建 reduced model，详见 `native_pinocchio_review.md`。
50 Hz 编码器 FK 日志不能排除更高频振动、减速器间隙和连杆柔性；末端上下抖动仍需外部观测确认。

## 已完成的验证

原生 Pinocchio 3.9.0 与 4.1.0 分别运行全部回归；包含 DM 两种控制模式的连续运动、原生重力、
到位阶段、旧缓存/单轴失联、故障状态、量程不匹配、COM/tty/cu 和只读预检。
原生 Rust 回归覆盖收发锁分离、完整报文不交错、批量编码与 SDK 单轴编码一致、
错误批次发送前检查、接收时间与序号、寄存器回复不刷新传感器时间，以及 RS 原有锁修复。
进一步优化后的完整回归：Pinocchio 3.9.0 / 4.1.0 各 111 项通过；原生 Rust 已验证共 53 项通过。
没有运行真机或在另外两个操作系统执行控制。

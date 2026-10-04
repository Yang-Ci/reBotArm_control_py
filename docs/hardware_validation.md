# RS 低速抖动：真机验证与从上到下复查

适用：RobStride、PCAN、`8_arm_traj_control.py`，保持姿态沿世界 Z 上升 10 cm / 8 s。
Windows、Ubuntu、macOS **分别独立运行**；任意一次运行都只开一个 PCAN 总线连接。
这里没有执行真机运动。源码回归通过不能替代本机摩擦、装配和负载的验收。
DM 串口桥与保留 500 Hz 的对应方案见 [dm_low_speed_motion_fix.md](dm_low_speed_motion_fix.md)。
周期、日志与重力计算的后续优化和调参顺序见 [进一步优化](control_timing_optimization.md)。

## 1. 本轮新增发现及修改

| 层次 | 原问题 | 本轮处理 |
| --- | --- | --- |
| 脚本 | 退出/中断后自动回零；末端状态使用全局默认模型 | 退出停止并失能，只有 `home` 显式回零；末端状态使用控制器实际硬件模型；`finally` 清理 |
| 生命周期 | 启动失败可能留下部分使能；组使能会影响同厂商其他组 | 启动异常按顺序停止线程、失能、关闭连接，保留原异常；每组只操作所属电机 |
| 到位 | 时间走完即完成，静止时不检查跟随误差 | 增加 settling 阶段，0.01 rad 内的新反馈持续 0.2 s 才完成，2 s 超时故障；静止也检查误差 |
| RS 反馈 | Python 线程分离后，SDK 原生句柄锁仍把读参数和 MIT 发送串行化 | 对 0.5.6 exact-host getter 克隆 Arc 后释放句柄锁；同一控制器上六关节并行读 mechPos |
| 采样 | 六关节顺序读取有时间差，失败批次可能被误认作新数据 | 整批成功才更新；记录请求开始跨度、整批耗时和序号；分析时剔除重复快照 |
| RS POS_VEL | 统一模式为 PP；CSP 便捷函数每次还会使能、重写模式 | 选择原生 CSP，启动配置模式/限速，循环只写 loc_ref；参数写入失败拒绝启动 |
| 周期 | SDK 多电机默认每帧 sleep 120 µs，在 Windows 可能放大 | RS 配置 `tx_gap_us: 0`，由 250 Hz 循环和 CAN 控制器调度；保留错误、超周期监测 |
| 诊断 | 难区分参考路径、时序和电机爬行 | 新增只读预检、静止记录、单关节探测、CSV 自动分析与状态 JSON |

先前已经修复的内容：固定 IK 容差吞掉小位移、目标位置阶梯和零速度、阻尼“零空间”泄漏、
解析 q/qd/qdd、路径/限位校验、独立 Pinocchio Data、公共动力学封装的缺失 API 与缓存问题。
详情见 `low_speed_motion_fix.md`、`native_pinocchio_review.md`。

源码证据：

- [0.5.6 参数读取 FFI](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_abi/src/motor_register_ffi.rs) 在等待回复期间持有 MotorHandle mutex。
- [0.5.6 模式与发送](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_vendors/robstride/src/motor.rs) 中 CSP helper 会调用 set_mode/enable。
- [0.5.6 PCAN 后端](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_core/src/pcan.rs) 为 Windows/macOS 使用 PCAN-Basic/PCBUSB，不能把第二个 Controller 当成独立接收订阅。
- [0.5.6 发送间隔](https://github.com/motorbridge/motorbridge/blob/v0.5.6/motor_core/src/controller.rs) 的多电机默认间隔为 120 µs。

以上是代码中可复现的问题；是否就是这台机械臂上下抖动的主因，仍需下列日志确认。

## 2. 环境与补丁安装

使用项目支持的 Python 3.10/3.11 和原生 Pinocchio。Windows 可以按
`native_pinocchio_review.md` 用 conda-forge 安装，Ubuntu/macOS 可使用已有原生环境。

在仓库根目录、准备运行真机的同一个 Python 环境执行：

```text
python -m pip install motorbridge==0.5.6 pytest
python tools/build_motorbridge_feedback.py
python -m pytest tests -q
```

构建需要 Rust 和平台 C 链接器：Windows 的 MSVC Build Tools、Ubuntu 的 build-essential、
macOS 的 Xcode Command Line Tools。该脚本下载固定 v0.5.6 源码并校验 SHA256，
先执行原生锁竞争回归，再生成 `.motorbridge/<平台>/` 下的库。不会替换已安装的 wheel。
本工作区已生成 Windows x64 库；换机器或平台需要重新构建。

重新启动 Python 后，RebotArm 自动选择对应本地库。若已有 `MOTORBRIDGE_LIB`，该显式设置
优先；预检会显示实际库路径。启动 RS 末端控制时会检查补丁标记，防止误用 stock 库。
只升级 wheel **不包含本地的锁修复**。

构建的库保留 RS/PCAN/SocketCAN 和 DM 串口桥路径，关闭另一条 DM_Device 专用 USB SDK：
0.5.6 打包的 DM C++ shim/header 在本机编译不匹配，关闭该路径还需补上其缺失的 cfg。
这里的 DM_Device 指 USB2CANFD 等专用 SDK 路径，不是 COM/tty/cu 串口桥。
如果要使用 DM_Device，应使用对应 SDK 构建。
macOS 的 PyPI 0.5.6 wheel 只有 Apple Silicon；Intel Mac 可以安装 sdist 的 Python 绑定，
再运行本地原生构建。

| 平台 | 通道与运行库 |
| --- | --- |
| Windows | `channel: can0` 对应 PCAN_USBBUS1；安装同架构 PEAK PCAN-Basic，确保 PCANBasic.dll 可找到 |
| Ubuntu | PCAN 使用内核 SocketCAN/peak_usb，配置好 1 Mbps 的 can0；此处不是字符设备 `/dev/pcan*` |
| macOS | `channel: can0` 对应 PCAN_USBBUS1；安装 SDK 所需的 MacCAN PCBUSB，使 libPCBUSB.dylib 可找到 |

示例 Ubuntu 配置命令（确认实际接口名和本机电机的波特率后执行）：

```bash
ip -details link show can0
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

当前 250 Hz 配置针对 1 Mbps；若用 500 kbps，不应直接沿用该发送频率。
SDK 允许显式环境变量 `MOTORBRIDGE_TX_GAP_US` 覆盖配置，预检会记录它。
各平台都需单独验收周期，Python/桌面系统没有硬实时保证。

## 3. 只读预检：先验证反馈

机械臂放在有支撑、关节零点正确、路径有空间的姿态，先不要启动运动脚本。
测试期间关闭其他控制程序；保留可操作的硬件急停。退出运动脚本会失能，因此要有支撑。

```text
python tools/rs_hardware_check.py --hw rebotarm_rs.yaml --seconds 10 --output logs/preflight.csv
```

工具只读 mechPos：不改模式、不使能、不清零，也不在退出时失能。
输出 CSV 与同名 JSON，检查：

- `motorbridge` 为 0.5.6、`feedback_lock_patch` 为 true，实际 ABI 路径正确。
- `ready_for_control` 为 true；至少有连续样本，没有读取错误，关节在 URDF 限位内。
- 关节名称、ID、角度方向、零点、URDF 与装配一致。限位内并不证明零点正确。
- 请求开始跨度应很小，整批耗时通常至少一个 SDK 8 ms 轮询间隔；明显超时或间歇停顿先查总线。
- 支撑状态下编码器不应大幅跳动；预检的 gravity 值只是模型计算，尚未证明实际力矩方向/负载正确。

请求开始时间、耗时均是本地时间。SDK 没有硬件采样时间戳，平行读取也不是严格同步采样。

## 4. 静止与单关节试验

保持 `friction.enabled: false`，使用当前 MIT 配置启动：

```text
python example/8_arm_traj_control.py --hw rebotarm_rs.yaml
```

先输入 `status`、`end_state`，确认机械臂稳定。输入：

```text
record
```

等待 10 s，然后：

```text
status
log logs/hold.csv
```

另一个终端分析日志（它只读文件）：

```text
python tools/analyze_motion_log.py logs/hold.csv --output logs/hold_report.json
```

每次 `log` 也保存 `.status.json`，包含全部伺服周期的 max_dt、超周期次数和反馈错误数。
故障先 `stop` 并保存日志；查明原因后再 `clear_fault`。`stop` 是软件保持，不是断电急停。
保持时已抖，应先检查重力、零点、负载和增益，不要先上摩擦补偿。

稳定后测试关节 2/3 的小幅往返，确认机械姿态和避障条件允许；以下角度是示例探测幅度：

```text
joint joint2 0.02 2
```

等待 `status` 显示 `moving: false, goal_reached: true, fault: None`，导出 `logs/joint2_up.csv`。
再执行 `joint joint2 -0.02 2`，同样等待到位并导出。关节 3 按相同方式测试。
每个新动作会开始新日志，上一条必须先导出。单关节轨迹没有笛卡尔碰撞规划。
单关节日志使用下列命令分析时序与关节误差，跳过竖直轨迹的单调 Z 判据：

```text
python tools/analyze_motion_log.py logs/joint2_up.csv --joint-only
```

## 5. 原始问题复现：由小到大

起始姿态应远离直臂奇异位形和关节限位，末端 Z 上下都有余量。
下面每一步独立发送；等待实测到位、保存日志后再发下一步，不要一次粘贴整组命令。

| 动作 | 上升命令 | 对应返回 | 日志示例 |
| --- | --- | --- | --- |
| 1 cm / 2 s | `lift 0.01 2` | `lift -0.01 2` | lift_1cm_2s.csv |
| 5 cm / 4 s | `lift 0.05 4` | `lift -0.05 4` | lift_5cm_4s.csv |
| 10 cm / 8 s | `lift 0.1 8` | `lift -0.1 8` | lift_10cm_8s.csv |

上升完成后输入：

```text
status
log logs/lift_10cm_8s.csv
```

分析原始问题：

```text
python tools/analyze_motion_log.py logs/lift_10cm_8s.csv --expected-dz 0.1 --output logs/lift_10cm_8s_report.json
```

默认分析门槛：Z 终点误差 ≤2 mm，反向回退 ≤0.5 mm，保持峰峰值 ≤1 mm，
无错误/故障，最大周期 ≤50 ms、反馈年龄 ≤250 ms、请求开始跨度 ≤20 ms。
这些是初次排障筛选标准，可用工具参数改变位置门槛，不是该机械臂精度承诺。
控制器到位依据关节误差；Z 精度仍由报告单独判断。
`control_dt` 的分位数只来自记录的周期，全部周期最大值以 `.status.json` 为准。
对于 250 Hz 的 4 ms 目标，应进一步检查 p95/p99 和超周期比例；即使未触发 50 ms 故障，
长期运行在更低频率也说明调度/传输未达到配置目标。

8 s 通过后，返回同一初始姿态，分别试 10 cm / 4 s、10 cm / 16 s。
每种速度做上下往返 3 次并分别保存；上升和下降要分别验收。
换平台时保持同一装配、负载、姿态与参数重新执行整套流程，比较日志。

## 6. 有抖动时怎样往下定位

| 观察 | 下一步 |
| --- | --- |
| z_ref 本身回退、位置跳变或 IK 失败 | 查目标姿态、奇异性、关节限位和模型；规划失败不会安装轨迹 |
| z_ref 正常，max_dt/反馈错误异常，某个平台更明显 | 核对补丁是否加载、PCAN 驱动/波特率、SDK TX 间隔与桌面调度；先处理通信 |
| 静止也抖，tau_ff 大或长期顶到限幅 | 查零点、力矩方向、重力方向、URDF 质量/质心和真实负载，再调整 Kp/Kd |
| 静止稳定，低速时某关节误差积累后突然跳，16 s 比 4 s 明显 | 符合黏滑/爬行特征；本机关节正反向辨识摩擦，逐步启用有界前馈后复测 |
| 只有反向时出现位移 | 查减速器齿隙、轴承、联轴器、线缆拖拽及方向相关摩擦 |
| 编码器 FK 平稳，肉眼末端仍抖 | 用固定机位视频/外部量具验证结构挠曲、齿隙与高频振动，不能从本日志排除 |

不要直接采用仓库其他机器的摩擦结果或先增大 Kp。先校准重力/负载，再做当前机器的低速往返辨识，
区分库仑摩擦与起动静摩擦，从小补偿比例开始。首次验收不建议运行会释放支撑力矩的 auto_float_test。

## 7. 当前验证范围

| 本轮验证 | 结果 |
| --- | --- |
| 原生 Pinocchio 3.9.0 / Python 3.11.16 / motorbridge 0.5.6 | 72 项通过 |
| 原生 Pinocchio 4.1.0 / Python 3.11.16 / motorbridge 0.5.6 | 72 项通过 |
| 显式 NumPy 替身 / Python 3.11.16 | 47 项通过，25 项原生验证跳过 |
| Windows 原生 Rust 同句柄并发回归 | 补丁通过；恢复上游 getter 后，同测试被阻塞 304.5 ms 并失败 |
| Windows release ABI、Python 符号绑定与补丁标记 | 通过 |
| Python 3.10 语法解析与四个 CLI 帮助入口 | 通过，未打开 CAN |

原生并发测试使用模拟 CAN 和人为设置的 300 ms 无回复超时，用于证明句柄锁会阻塞发送；
不是实机周期测量。当前反馈实际请求超时配置为 20 ms。
Ubuntu/macOS 共用同一 Arc/锁修复源码，尚未在这两台实际平台编译/连接验收。
PCAN 收发故障时 SDK 的重连仍会耗时，软件停机不能替代硬件急停。

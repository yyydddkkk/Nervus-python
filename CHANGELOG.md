# 变更记录

## 0.1.0a1 — 开发者预览

- 持久 Python Session，支持顶层 await 和跨 Turn 状态。
- Turn 级能力快照、完整发布及显式局部更新。
- 主动停止、超时处理和明确的 Worker 状态丢失反馈；不提供自动恢复。
- 受管外部程序与有界标准输出、标准错误。
- ScriptedModel、DeepSeek Adapter 和终端 Agent。
- 四个无需模型 API 的使用示例，以及 102 项正式测试。

仅支持 Linux / CPython 3.14.x。API 尚不稳定。使用方式与限制见 [README](README.md)。

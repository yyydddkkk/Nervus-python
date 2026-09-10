# Nervus Python

[English](README.md) | **简体中文**

**0.1.0a1 · 开发者预览版 · 尚未公开发布到 PyPI。**

代码驱动的 Python Agent Kernel，提供可嵌入的 Host API 和终端 Agent。同一 Session 保留 Python 变量、函数和结果。公开 API 仍可能变化。

仅支持 **Linux / CPython 3.14.x**，运行依赖仅标准库。Windows、macOS 和其他 Python 版本未验证。

## 安装与示例

从源码目录安装：

```sh
uv sync --locked
uv run --locked python examples/basic_session.py
uv run --locked python examples/scripted_session.py
uv run --locked python examples/host_session.py
uv run --locked python examples/partial_update.py
```

示例：[基本 Session](examples/basic_session.py)、[脚本模型](examples/scripted_session.py)、[Host 与受管进程](examples/host_session.py)、[局部能力更新](examples/partial_update.py)。四个示例不调用模型 API；Host 示例会启动并停止本地 Python 子进程。

## 终端 Agent

```sh
uv run --locked nervus --cwd /absolute/project --env-file /absolute/path/to/.env --show-python
```

也可设置 `DEEPSEEK_API_KEY` 并省略 `--env-file`。不会自动读取目标项目的 `.env`。默认模型为 `deepseek-v4-flash`；提交任务后会请求模型 API，可能产生费用。

- `/stop` 或 Ctrl+C：停止当前 Turn。
- `/workspace`：空闲时查看工作变量元数据。
- `/history`、`/status`：查看聊天与状态；`/clear-history` 只清聊天。
- `/new`：关闭旧 Session，清空聊天和变量，不删除已写文件。
- `/quit` 或 EOF：清理并退出。
- `--show-python`：显示代码和执行反馈；`--debug`：显示完整调试细节，注意敏感信息。

每行输入对应一个 Turn，忙时不排队。历史与模型上下文有大小限制，超限暂停，不静默裁剪。聊天不跨终端重启保留。

## 基本 Host 调用

```python
from nervus import Session, Code, Finish, ScriptedModel


def main():
    with Session() as session:
        result = session.run('计算并保存', ScriptedModel([
            Code('saved = 6 * 7', exports=('saved',)),
            Finish('完成'),
        ]))
        print(result.feedback[0].values)  # {'saved': 42}


if __name__ == '__main__':
    main()
```

Worker 使用 spawn 创建。Host 脚本必须保护入口，能力工厂必须可被 spawn 导入。能力注册与更新用法见 [Host 示例](examples/host_session.py) 和 [局部更新示例](examples/partial_update.py)。

## 限制与安全

- **只执行可信代码，不是沙箱。** 文件、网络及其他宿主副作用由应用方负责隔离。
- 代码异常不会回滚变量或文件。Worker 丢失后没有自动恢复；终端需显式 `/new`。
- 停止不能强制取消任意同步模型调用，也不能管理任意用户线程或绕过受管接口启动的进程。
- 受管进程使用 POSIX 进程组，不覆盖主动脱离的程序，不保证监督端崩溃后的回收。
- 本项目不承诺真实模型任务成功率或生产级可靠性。

## 验证与参与

```sh
uv run --locked --offline python -m unittest discover -s tests -t . -v
uv build --offline
```

先运行 `uv sync --locked` 准备环境。当前正式测试基线为 102 项，不需要模型 API 密钥。CI 在 Linux / Python 3.14 上运行这套测试并构建 wheel/sdist；uv 离线模式不是网络隔离。

- [贡献说明](CONTRIBUTING.md)
- [安全与漏洞报告](SECURITY.md)
- [变更记录](CHANGELOG.md)

代码采用 [MIT License](LICENSE)。

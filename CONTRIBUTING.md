# 参与贡献

本项目是 Linux / CPython 3.14.x 开发者预览版。使用方式与限制见 [README](README.md)。

## 本地验证

安装 uv，在源码目录运行：

```sh
uv sync --locked
uv run --locked --offline python -m unittest discover -s tests -t . -v
uv build --offline
```

首次准备工具和构建依赖需要联网。正式测试基线为 102 项，不需要真实 API 密钥。常规验证不要请求付费模型 API。

验证公开副本及分发产物：

```sh
uv run --locked python scripts/verify_public.py --out-dir /absolute/path/to/new-output
```

`--out-dir` 必须是尚不存在的新目录。该[验证脚本](scripts/verify_public.py)通过[公开导出脚本](scripts/export_public.py)创建不含内部资料的副本，检查构建、安装、四个示例和正式测试，并保存产物与验证报告。准备构建依赖可能需要联网；不会调用模型 API。


## 提交变更

- 报告问题时提供系统、Python 版本、最小复现和预期/实际结果。
- 大范围 API 变更先讨论。保持补丁聚焦，为行为变化添加测试。
- PR 写明变更目的、验证命令、结果和已知限制，并更新相关公开说明。
- 不提交 `.env`、密钥、用户数据、敏感日志或本地缓存。
- 安全问题按 [SECURITY.md](SECURITY.md) 处理。

贡献采用本项目的 [MIT License](LICENSE)。请确认你有权提交代码和材料，并保留适用的第三方版权及许可证声明。

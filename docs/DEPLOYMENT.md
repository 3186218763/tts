# 本地运行

当前训练好的 v2Pro 权重和参考音频位于项目的 `model/` 目录（权重由 Git LFS
管理），项目提供两个进程：GPT-SoVITS TTS API 和本项目 Web 对话层。DeepSeek API key 仍需写入
被 gitignore 的 `configs/config.yaml`。

先启动 TTS API（使用安装了 GPT-SoVITS 依赖的 Python 环境）：

    python scripts/run_huayin_api.py --python /home/mtr/miniconda3/envs/gptsovits/bin/python

然后启动 Web（web extra 同时安装 faster-whisper）：

    pip install -e '.[web]'
    python -m frontend.web --host 127.0.0.1 --port 8000

浏览器打开 http://127.0.0.1:8000/。健康检查为
http://127.0.0.1:8000/healthz。

局域网访问时，将 Web 监听地址改为所有网卡，并使用本机局域网 IP：

    python -m frontend.web --host 0.0.0.0 --port 8001

例如本机 IP 为 `192.168.1.62` 时，其他同一局域网设备打开
`http://192.168.1.62:8001/`。只需暴露 Web 端口；GPT-SoVITS API 默认继续绑定
`127.0.0.1:9880`，由 Web 后端在本机访问。

健康检查会同时报告配置和运行时状态：`llm_configured` 需要真实的
DeepSeek key，`tts_available` 表示 GPT-SoVITS API 当前可访问，
`asr_available` 表示 faster-whisper 依赖可以加载。示例：

    {
      "status": "ok",
      "llm_configured": true,
      "tts_configured": true,
      "tts_available": true,
      "asr_configured": true,
      "asr_available": true
    }

CLI 启动时也会先探测 TTS API；服务未启动会直接提示启动命令，不会进入一个
必然失败的对话循环。LLM 首次请求在尚未收到 token 时会自动重试一次；单句
TTS 失败只跳过该句音频，文字回复和后续句子仍会继续。

如果不需要浏览器，可在 TTS API 已启动且配置文件已填写后运行：

    python -m frontend.cli

scripts/run_huayin_api.py 默认加载项目 `model/` 下的 huayin-e15.ckpt 和
huayin_e8_s2536.pth，也可以用 --gpt-model、--sovits-model 选择其他 epoch。
Web 层通过 SSE 推送
sentence、audio、done 事件，浏览器会按顺序显示文字并提供 WAV 播放控件。

录音按钮使用本地 faster-whisper。ASR 模型、设备和上传大小可在
configs/config.yaml 的 asr 节调整；第一次转写会下载并加载模型，健康检查中的
asr_available 字段可确认依赖是否就绪。

若只做一次本地模型检查，不启动 API：

    python scripts/test_huayin_tts.py "你好，今天也要加油。" --dry-run
    python scripts/test_huayin_tts.py "你好，今天也要加油。"

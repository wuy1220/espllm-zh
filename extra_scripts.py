"""PlatformIO 构建前闸门：src/model_weights.hpp 必须自洽（audit 19.1）。

为什么只做「头文件自洽」而不是「头文件 vs checkpoint」：SCons 用的是 PlatformIO 自带的
Python，未必装了 torch；在这里 import torch 会把构建变成环境依赖问题。
完整的跨文件比对请在项目虚拟环境里跑 `python check_arch_sync.py`（会读 __arch__）。

闸门能拦住的是最常见的那种漂移：手改了头文件常量、或者用旧版转换器导出的头文件
（没有 arch 注释 / 没有 model_newline_id）—— 这些以前都能一路编译通过并烧进板子。
"""
import os
import subprocess
import sys

Import("env")  # noqa: F821

PROJ = env.subst("$PROJECT_DIR")  # noqa: F821
GATE = os.path.join(PROJ, "check_arch_sync.py")
PY = os.environ.get("ESPGPT_PYTHON", sys.executable)

if os.path.exists(GATE):
    rc = subprocess.run([PY, GATE, "--header-only"], cwd=PROJ).returncode
    if rc != 0:
        sys.stderr.write(
            "\n[check_arch_sync] 权重头文件与它自报的架构不一致，拒绝构建。\n"
            "  重新导出： ESPGPT_CHAR_VOCAB=persona/vocab_zh.json "
            "python convert_model_to_c.py <checkpoint-name>\n"
        )
        env.Exit(1)  # noqa: F821
else:
    sys.stderr.write("[check_arch_sync] 未找到 check_arch_sync.py，跳过架构闸门\n")

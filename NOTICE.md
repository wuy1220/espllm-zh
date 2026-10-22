# 来源与许可范围

## 上游

推理引擎的骨架来自 [ahmedbarakat207/espllm](https://github.com/ahmedbarakat207/espllm)：

- `src/inference.hpp`、`src/main.cpp` 的原有结构
- 权重打包格式（ternary 1.58-bit、int8 词表、fp16 scale）
- `platformio.ini`、`partitions*.csv`、烧录脚本

**上游仓库未声明任何开源许可证。** 复用这部分代码前，请自行联系上游作者确认授权。
本仓库不因下文的 MIT 许可证而改变上游对这些文件的权利。

## 本项目新增

以下内容由本项目编写，以 [MIT](LICENSE) 发布：

- 中文人设层：`persona/vocab_zh.json`、`persona/stance_lexicon.py`、`gen_*_corpus.py`
- 意图规则层：`intent_rules.py`、`gen_rules_cpp.py`、`src/intent_rules_gen.h`
- 训练与导出：`main.py` 的改动、`convert_model_to_c.py`、`check_arch_sync.py`
- 质量工具：`probe_stance.py`、`per_concept.py`、`board_stance4.py`、`probe_board_acceptance.py`、
  `check_lexicon.py`、`coverage_zh_rules.py`、`test_newline_token.py`
- 全部文档：`docs/`、`persona/*.md`

## 同人内容

角色「琪露诺」及《东方 Project》相关设定归上海爱丽丝幻乐团（ZUN）所有。
本仓库不含原作素材，人设台词为二次创作，仅用于技术演示。

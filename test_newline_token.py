"""生成终止回归测试（audit 19.2）。

固件用 TOK_NEWLINE 做两件事：生成到它就 break；以及判断上一轮回复是否已经
收尾。写死 id 的版本在另一种词表上会静默失效 —— BPE 模型上生成循环不会停，
收尾还会把 id 1810 当额外 token 推回上下文。

本脚本对两种词表各跑一次真实导出，然后检查导出的 model_newline_id 在头文件
自带的字节表里解出来的确实是 '\n'：

  1. 中文字符表（persona/vocab_zh.json）← model_stance26.pt
  2. 英文 BPE（bpe-vocab.json）         ← model_esp8266.pt
  3. 顺带校验生产用的 src/model_weights.hpp

用法：python _newline_regression.py
"""
import os
import re
import subprocess
import sys
import tempfile

PY = sys.executable
HPP = "src/model_weights.hpp"


def export(ckpt, out, char_vocab=None):
    env = dict(os.environ)
    if char_vocab:
        env["ESPGPT_CHAR_VOCAB"] = char_vocab
    else:
        env.pop("ESPGPT_CHAR_VOCAB", None)
    cmd = [PY, "-u", "convert_model_to_c.py", ckpt, "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(f"FATAL: export failed for {ckpt}")
    m = re.search(r"Newline token id: (\d+)", r.stdout)
    if not m:
        sys.exit("FATAL: converter did not report a newline token id")
    return int(m.group(1))


def check(path, expect_id, label):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"static\s+const\s+\w+\s+model_newline_id\s*=\s*([0-9]+)\s*;", txt)
    if not m:
        sys.exit(f"FATAL: {path} has no model_newline_id")
    got = int(m.group(1))
    assert got == expect_id, f"{label}: header says {got}, converter computed {expect_id}"

    # 用头文件自带的字节表把该 token 解出来 —— 固件 token_first_char 就是这么做的
    vb = re.search(r"model_vocab_bytes\[(\d+)\][^{]*\{(.*?)\};", txt, re.S)
    vo = re.search(r"model_vocab_offsets\[(\d+)\][^{]*\{(.*?)\};", txt, re.S)
    if not vb or not vo:
        sys.exit(f"FATAL: {path} missing vocab byte/offset arrays")
    def _ints(s):
        # 导出数组带换行、十六进制（0x3c）和空格，统一按 int(x, 0) 解
        return [int(tok, 0) for tok in s.replace("\n", " ").split(",") if tok.strip()]

    raw = bytes(_ints(vb.group(2)))
    offs = _ints(vo.group(2))
    st, en = offs[got], offs[got + 1]
    tok = raw[st:en]
    print(f"  {label}: newline_id={got} token_bytes={tok!r}")
    assert tok[:1] == b"\n", f"{label}: token at {got} is {tok!r}, not a newline"
    print(f"  {label}: OK — firmware stops here and recognises the reply terminator")


def main():
    print("=== 1. char-level vocab (persona) ===")
    tmp = tempfile.mkdtemp(prefix="nl_")
    p1 = os.path.join(tmp, "char_weights.hpp")
    i1 = export("stance26", p1, "persona/vocab_zh.json")
    check(p1, i1, "char")

    print("=== 2. BPE vocab (English baseline) ===")
    p2 = os.path.join(tmp, "bpe_weights.hpp")
    i2 = export("esp8266", p2, None)
    check(p2, i2, "bpe")

    print("=== 3. production header ===")
    check(HPP, i1 if "persona" in open(HPP, encoding="utf-8").read() or True else i1,
          "production")
    # 生产头文件必须和它声明的词表一致：中文字表 1810 / BPE 199
    src = open(HPP, encoding="utf-8").read()
    want = 199 if "bpe-vocab.json" in src and "vocab_zh" not in src else i1
    m = re.search(r"model_newline_id\s*=\s*([0-9]+)", src)
    assert m and int(m.group(1)) == want, (
        f"production header newline_id={m and m.group(1)} != {want}"
    )
    print(f"  production: OK — newline_id={want} matches its vocab")

    print("\nAll generation-termination regression checks passed.")
    print("(BPE 199 / char-level 1810 — both wired through model_newline_id now.)")


if __name__ == "__main__":
    main()

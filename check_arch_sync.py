"""校验 src/model_weights.hpp 与 checkpoint 架构一致（audit 19.1）。

背景：固件只认头文件里的常量，训练只认 checkpoint 里的 __arch__，两者此前
没有任何关联。model_esp8266.pt（2048 词表 / 4L / 20 专家 / block 64）和
5L/32E/1850 的头文件可以相安无事地共存，谁用错 checkpoint 重新导出一次，
板上跑的模型就悄悄换掉了，而且编译照常通过。

用法：
    python check_arch_sync.py                 # 按头文件 arch-source 注释找 checkpoint
    python check_arch_sync.py model/model_stance26.pt
    python check_arch_sync.py --ckpt model/model_zh_canon4.pt

一致时退出码 0，不一致时退出码 1 并列出差异。建议在烧录/发布前跑一次。
"""
import argparse
import os
import re
import sys

HPP = "src/model_weights.hpp"
# 头文件常量名 -> checkpoint __arch__ 键名
FIELDS = {
    "model_vocab_size": "vocab_size",
    "model_n_embd": "n_embd",
    "model_n_layer": "n_layer",
    "model_n_head": "n_head",
    "model_n_kv_head": "n_kv_head",
    "model_block_size": "block_size",
    "model_group_size": "group_size",
    "model_mlp_hidden": "mlp_hidden",
    "model_n_experts": "n_experts",
}

# `// arch:` 注释里用的是短名（vocab=1850），与 __arch__ 的键名不同
COMMENT_ALIAS = {"vocab_size": "vocab"}


def parse_header(path):
    if not os.path.exists(path):
        sys.exit(f"FATAL: {path} not found (run convert_model_to_c.py first)")
    txt = open(path, encoding="utf-8").read()
    src = None
    m = re.search(r"^// arch-source:\s*(\S+)", txt, re.M)
    if m:
        src = m.group(1)
    consts = {}
    for name in FIELDS:
        m = re.search(r"static\s+const\s+\w+\s+%s\s*=\s*([0-9]+)\s*;" % name, txt)
        if not m:
            sys.exit(f"FATAL: {name} not found in {path} — header format changed?")
        consts[name] = int(m.group(1))
    m = re.search(r"static\s+const\s+\w+\s+model_newline_id\s*=\s*([0-9]+)\s*;", txt)
    consts["model_newline_id"] = int(m.group(1)) if m else None
    return src, consts


def load_arch(path):
    import torch

    if not os.path.exists(path):
        sys.exit(f"FATAL: checkpoint {path} not found")

    # .quantized 存的是量化后的模型对象，反序列化它要把 main.Transformer 挂进
    # __main__（convert_model_to_c.py 就这么干），很重。量化产物总是和同名的
    # raw checkpoint 成对出现，而 raw 那个一定带 __arch__ —— 先走它。
    if path.endswith(".quantized"):
        sib = path[: -len(".quantized")]
        if sib != path and os.path.exists(sib):
            print(f"  note: {os.path.basename(path)} is a quantized model object;")
            print(f"        read __arch__ from its raw sibling {os.path.basename(sib)}")
            path = sib

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "__arch__" in obj:
        return obj["__arch__"]

    # 没有 __arch__ 的老 checkpoint：改用 tensor 形状反推（与转换器同源），
    # 至少能抓住 vocab / 层数 / 专家数这一类会改变固件行为的差异。
    sd = obj

    def _rows(key):
        for k in (key + ".weight", key + ".qweight"):
            if k in sd:
                return sd[k].shape
        return None

    tok = _rows("tok_emb")
    if tok is None:
        sys.exit(f"FATAL: cannot find tok_emb in {path}")
    qkv_out, _ = sd["blocks.0.attn.qkv.weight"].shape
    embd = tok[1]
    n_layer = len(
        [k for k in sd if k.startswith("blocks.") and k.endswith(".attn.qkv.weight")]
    )
    n_experts = len(
        [
            k
            for k in sd
            if k.startswith("blocks.0.mlp.experts.") and k.endswith(".gate_proj.weight")
        ]
    )
    gate_out, _ = sd["blocks.0.mlp.experts.0.gate_proj.weight"].shape
    block = sd["rope_cos"].shape[0] if "rope_cos" in sd else None
    print(f"  note: {path} has no __arch__ — derived arch from tensor shapes")
    return {
        "vocab_size": int(tok[0]),
        "n_embd": int(embd),
        "n_layer": int(n_layer),
        "n_head": None,
        "n_kv_head": None,
        "block_size": int(block) if block else None,
        "group_size": None,
        "mlp_hidden": int(gate_out),
        "n_experts": int(n_experts),
    }


def parse_header_arch_comment(path):
    """读 `// arch: k=v k=v …` 注释，返回 dict。转换器每次导出都会写这行。"""
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"^// arch:\s*(.+)$", txt, re.M)
    if not m:
        return None
    out = {}
    for kv in m.group(1).split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = int(v) if v.isdigit() else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=None)
    ap.add_argument("--ckpt", dest="ckpt_opt", default=None)
    ap.add_argument("--hpp", default=HPP)
    ap.add_argument(
        "--header-only",
        action="store_true",
        help="只校验头文件自洽（常量 == arch 注释），不加载 checkpoint，无需 torch。"
             "供 PlatformIO 构建闸门使用。",
    )
    a = ap.parse_args()

    src, consts = parse_header(a.hpp)

    if a.header_only:
        claimed = parse_header_arch_comment(a.hpp)
        if claimed is None:
            print(f"MISMATCH — {a.hpp} 里没有 `// arch:` 注释行，无法自洽校验。")
            print("  重新导出：ESPGPT_CHAR_VOCAB=persona/vocab_zh.json python convert_model_to_c.py <name>")
            sys.exit(1)
        bad = []
        for hname, akey in FIELDS.items():
            hv = consts[hname]
            av = claimed.get(akey, claimed.get(COMMENT_ALIAS.get(akey, akey)))
            if av is None:
                bad.append(f"  {akey}: 注释里没有这一项")
            elif hv != av:
                bad.append(f"  {akey}: 常量={hv} vs 注释={av}")
        if consts.get("model_newline_id") is None:
            bad.append("  model_newline_id: 缺失（换行 token 会退回硬编码）")
        if bad:
            print(f"MISMATCH — {a.hpp} 的常量与它自报的架构不一致：")
            print("\n".join(bad))
            sys.exit(1)
        print(f"header-only OK — {a.hpp} 自洽（来源 {src or '未知'}）")
        return

    ckpt = a.ckpt_opt or a.ckpt or src
    if not ckpt:
        sys.exit(
            "FATAL: header carries no arch-source comment and no checkpoint given.\n"
            "       Re-export with convert_model_to_c.py, or pass the checkpoint:\n"
            "       python check_arch_sync.py model/model_stance26.pt"
        )

    print(f"header : {a.hpp}")
    print(f"         " + " ".join(f"{k[len('model_'):]}={v}" for k, v in consts.items()))
    print(f"ckpt   : {ckpt}")
    arch = load_arch(ckpt)
    print(f"         " + " ".join(f"{k}={v}" for k, v in sorted(arch.items())))

    bad = []
    for hname, akey in FIELDS.items():
        hv, av = consts[hname], arch.get(akey)
        if av is None:
            continue  # 老 checkpoint 推不出来的字段（n_head 等），跳过
        if hv != av:
            bad.append(f"  {akey}: header={hv} vs ckpt={av}")
    if bad:
        print("\nMISMATCH — the header does not describe this checkpoint:")
        print("\n".join(bad))
        print(
            "\nThe firmware would run a different model than this checkpoint.\n"
            "Either re-export from the intended checkpoint, or point the header's\n"
            "arch-source at the one it was actually generated from."
        )
        sys.exit(1)

    if consts.get("model_newline_id") is None:
        print("\nWARNING: header has no model_newline_id — regenerate with a current convert_model_to_c.py")
        sys.exit(1)

    print("\nOK — header architecture matches the checkpoint.")


if __name__ == "__main__":
    main()

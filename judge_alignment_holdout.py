"""Blindly compare control and alignment A/B replies on the sealed holdout."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import generate_alignment_distill as generation


DEFAULT_DIR = Path(r"E:\espllm_data\persona_corpus\alignment_distill_ab_20261024")


def build_prompt(batch):
    cases = []
    for row in batch:
        cases.append(
            f"{row['id']}. 用户：{row['u']}\nA：{row['A']}\nB：{row['B']}"
        )
    return """你是中文封闭域角色扮演对话的质量评审。角色是琪露诺，但请优先看用户实际体验。
逐题比较 A/B：回复是否直接处理用户当前问题，是否利用了用户给出的细节，回答形式是否合适，是否保持自然角色口吻。不要偏爱更长的回答；不要按字面匹配某个参考答案。两个都不切题或明显不合格时选 both_bad，质量接近时选 tie。

只输出 JSON：{"rows":[{"id":1,"preferred":"A|B|tie|both_bad","A":{"relevance":0,"details":0,"form":0,"persona":0},"B":{"relevance":0,"details":0,"form":0,"persona":0},"reason":"不超过20字"}]}
各维度打 0-2 分：0=明显失败，1=部分做到，2=做到。不要推测隐藏标签。

题目：
""" + "\n\n".join(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--predictions", default="holdout_predictions.jsonl")
    parser.add_argument("--control", default="control.pt.best")
    parser.add_argument("--candidate", default="alignment.pt.best")
    parser.add_argument("--output", default="holdout_judgments.jsonl")
    args = parser.parse_args()

    generation.load_env()
    base, key = os.environ.get("QN_BASE", ""), os.environ.get("QN_KEY", "")
    model = os.environ.get("QN_MODEL", "")
    if not (base and key and model):
        raise SystemExit("QN_BASE, QN_KEY, and QN_MODEL must be set")

    path = Path(args.predictions)
    if not path.is_absolute():
        path = args.directory / path
    with path.open(encoding="utf-8") as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    rng = random.Random(args.seed)
    cases = []
    for index, row in enumerate(source, 1):
        control = row["predictions"][args.control]
        candidate = row["predictions"][args.candidate]
        swap = bool(rng.getrandbits(1))
        cases.append({
            "id": index,
            "u": row["u"],
            "A": candidate if swap else control,
            "B": control if swap else candidate,
            "candidate_side": "A" if swap else "B",
        })

    # Resume: a rate-limited run leaves <output>.partial.jsonl behind, and
    # re-judging all 64 items would hit the same RPM wall forever.
    results, errors = [], []
    partial = args.directory / (Path(args.output).stem + ".partial.jsonl")
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                results.append(item)
        done_ids = {int(item["id"]) for item in results}
        cases = [case for case in cases if case["id"] not in done_ids]
        print(f"resume: {len(results)} already judged, {len(cases)} to go", flush=True)

    for start in range(0, len(cases), args.batch_size):
        batch = cases[start:start + args.batch_size]
        try:
            parsed = elapsed = usage = None
            for attempt in range(6):
                try:
                    parsed, elapsed, usage = generation.api_request(
                        build_prompt(batch), base, key, model, timeout=180
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    if attempt == 5 or not any(
                            token in msg for token in ("429", "502", "503", "504", "timed out")):
                        raise
                    wait = 15 * (attempt + 1)
                    print(f"  retry {attempt + 1} after {wait}s: {msg[:90]}", flush=True)
                    time.sleep(wait)
            returned = parsed.get("rows", [])
            if len(returned) != len(batch):
                raise ValueError(f"expected {len(batch)} judgments, got {len(returned)}")
            by_id = {int(item["id"]): item for item in returned}
            if any(case["id"] not in by_id for case in batch):
                raise ValueError("judge response ids do not match request")
            for case in batch:
                item = by_id[case["id"]]
                item["candidate_side"] = case["candidate_side"]
                item["control_side"] = "B" if case["candidate_side"] == "A" else "A"
                item["u"] = case["u"]
                item["A_text"], item["B_text"] = case["A"], case["B"]
                results.append(item)
            print(f"judged {len(results)}/{len(cases)} in {elapsed:.1f}s; usage={usage}", flush=True)
        except Exception as exc:
            errors.append({"ids": [case["id"] for case in batch], "error": str(exc)})
            print(f"batch {start // args.batch_size + 1} failed: {exc}", flush=True)
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            for item in results:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    out = Path(args.output)
    if not out.is_absolute():
        out = args.directory / out
    if not errors:
        out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
                       encoding="utf-8")
    print(json.dumps({"judged": len(results), "expected": len(cases), "errors": errors,
                      "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

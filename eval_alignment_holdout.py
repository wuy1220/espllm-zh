"""Generate matched greedy replies on the sealed alignment holdout."""
import argparse
import collections
import json
import os
from pathlib import Path


DEFAULT_DIR = Path(r"E:\espllm_data\persona_corpus\alignment_distill_ab_20261024")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--checkpoints", nargs="+", default=["control.pt.best", "alignment.pt.best"])
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--output", default="holdout_predictions.jsonl")
    args = parser.parse_args()

    holdout_path = args.directory / "alignment_holdout.jsonl"
    with holdout_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    import gen_long

    outputs = {name: [""] * len(rows) for name in args.checkpoints}
    for name in args.checkpoints:
        gen_long._MAIN = None
        gen_long._MODEL = None
        checkpoint = args.directory / name
        gen_long.load(str(checkpoint), threads=8)
        for index, row in enumerate(rows):
            prompt = f"User: {row['u']}\nBot:"
            _, token_ids, _ = gen_long.run(
                prompt, win=104, maxnew=args.max_new, seed=20261024 + index,
                temp=0.0, topk=1, rep=1.6, repwin=20,
            )
            text = gen_long.dec(token_ids).split("\n", 1)[0].strip()
            if text.lower().startswith("bot:"):
                text = text[4:].strip()
            tags = os.environ.get("ESPGPT_PLAN_TAGS", "")
            if tags and text[:1] in tags:
                text = text[1:].lstrip()
            outputs[name][index] = text
        print(f"generated {len(rows)} holdout replies with {checkpoint}", flush=True)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = args.directory / out_path
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            result = {**row, "predictions": {name: outputs[name][index] for name in args.checkpoints}}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    for name, values in outputs.items():
        nonempty = [value for value in values if value]
        print(json.dumps({
            "checkpoint": name,
            "total": len(values),
            "nonempty": len(nonempty),
            "mean_chars": round(sum(len(value) for value in values) / max(1, len(values)), 2),
            "unique_replies": len(set(nonempty)),
            "task_counts": dict(collections.Counter(row["kind"] for row in rows)),
        }, ensure_ascii=False))
    print(f"predictions: {out_path}")


if __name__ == "__main__":
    main()

"""Generate problem-aligned, non-template Cirno dialogue distillation data."""
import argparse
import collections
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENV_PATH = Path.home() / ".espllm" / "api.env"
VOCAB_PATH = HERE / "persona" / "vocab_zh.json"
TOPICS = [
    "青蛙", "冰与冷气", "雾之湖", "冰屋", "大妖精", "妖精同伴", "恶作剧",
    "算数", "最强宣言", "和灵梦比试", "和魔理沙比试", "夏天好热", "冬天最好",
    "雪人冰雕", "湖里的鱼", "冰符弹幕", "被小看", "迷路", "红魔馆", "⑨这个外号",
    "自我介绍", "打招呼", "喜好",
]
TASKS = [
    "回答一个具体事实问题，给出问题需要的信息，不绕开问题。",
    "解释眼前现象，并把原因说清楚；只解释用户问到的部分。",
    "针对用户提出的小麻烦给一个能执行的办法，不要泛泛安慰。",
    "回应用户的经历或心情，先接住对方，再顺着情境说一句角色化的话。",
    "面对质疑或调侃，直接回应其中的观点，保持自信但不要攻击用户。",
    "用户提出邀请或挑战时，明确接受、拒绝或还价，并说明缘由。",
    "用户说错或误解了幻想乡里的事情时，纠正具体误解，不要改讲别的话题。",
    "用户问一个需要选择的问题时，明确选项并给出符合角色的理由。",
    "用户给了具体细节或条件时，回复必须利用至少一个细节，不可退回通用台词。",
    "用户问到她确实不熟悉的域外事物时，坦率说不懂，再用自然的角色口吻收住。",
]
HAN_RE = re.compile(r"[\u3400-\u9fff]")
PUNCT = "。！？!?"


def load_env():
    if not ENV_PATH.is_file():
        raise SystemExit(f"Missing API config: {ENV_PATH}")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def norm(text):
    return re.sub(r"[\s\W_]+", "", text).lower()


def load_existing(paths):
    seen_questions, seen_answers = set(), set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("u"):
                    seen_questions.add(norm(row["u"]))
                if row.get("a"):
                    seen_answers.add(norm(row["a"]))
    return seen_questions, seen_answers


def api_request(prompt, base, key, model, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # QN_TEMPERATURE exists because some endpoints refuse anything but 1
        # (moonshotai/kimi-k3 answers 400 to 0.9), and re-running the judge on
        # a retired model is not an option.
        "temperature": float(os.environ.get("QN_TEMPERATURE", "0.9")),
        "max_tokens": int(os.environ.get("QN_MAX_TOKENS", "3000")),
        "reasoning_effort": os.environ.get("QN_REASONING", "low"),
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error", {}).get("message", "request rejected")
        except json.JSONDecodeError:
            detail = "request rejected"
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choice = result["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    decoder = json.JSONDecoder()
    parsed = None
    error = None
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            parsed = value
            break
    if parsed is None:
        diagnostic = {"finish_reason": choice.get("finish_reason"),
                      "message_fields": sorted(message), "usage": result.get("usage", {})}
        detail = "no JSON rows; " + json.dumps(diagnostic, ensure_ascii=False)
        if error:
            start = max(0, error.pos - 100)
            detail += "; near=" + repr(text[start:error.pos + 100])
        raise ValueError(detail)
    return parsed, time.time() - started, result.get("usage", {})


def build_prompt(cards):
    cases = "\n".join(
        f'{i + 1}. 主题：{card["topic"]}；互动任务：{card["task"]}'
        for i, card in enumerate(cards)
    )
    return f"""请为中文封闭域角色扮演模型生成训练对话。角色是《东方Project》的琪露诺：冰之妖精，住在雾之湖边的冰屋，自信、好胜、天真，有时犯傻；喜欢冬天和冰，夏天怕热，会把青蛙冻住再放回湖里；和大妖精亲近，输给灵梦或魔理沙也不服。

对每个指定互动任务，创作一条自然、具体的用户发言和琪露诺回复。用户只看到自己的发言，不会看到主题标签或任务说明。回复必须针对用户实际问的内容，优先使用用户给出的具体人物、物件、条件或情绪；不要用无关的熟悉台词顶替答案，也不要复述问题凑长度。根据问题决定回答结构：可以直接给结论、解释原因、给步骤、纠正误解、回应情绪、接受或拒绝邀请。句式和开头自然变化，不要每句都自称“咱”，但角色自称用“咱”或省略主语，不用“我”。回复一般 8-42 个汉字，最多两句，口语简短，不用比喻、成语、脏话、威胁或 AI 元话语。保留琪露诺的性格；算错可以偶尔发生，但不能把每道算术题都答错。使用常见中文字符，避免生僻字。

只输出 JSON 对象，格式为 {{"rows":[{{"t":"主题","kind":"任务类别","u":"用户发言","a":"角色回复"}}]}}，不要 Markdown 或解释。kind 用“事实、解释、办法、情绪、质疑、邀请、纠错、选择、细节、域外”之一。

逐项生成：
{cases}"""


def add_oov_guidance(prompt, banned_chars):
    if not banned_chars:
        return prompt
    return prompt + "\n不要使用这些字符：" + "".join(sorted(banned_chars))


def validate(row, card, seen_q, seen_a, vocab):
    if not isinstance(row, dict):
        return "not_object"
    if row.get("t") != card["topic"]:
        return "topic_mismatch"
    question, answer = str(row.get("u", "")).strip(), str(row.get("a", "")).strip()
    if not question or not answer:
        return "empty"
    if norm(question) in seen_q:
        return "duplicate_question"
    if norm(answer) in seen_a:
        return "duplicate_answer"
    if len(HAN_RE.findall(question)) > 35 or len(HAN_RE.findall(answer)) not in range(8, 43):
        return "length"
    if answer[-1] not in PUNCT:
        return "punctuation"
    if "我" in answer:
        return "wrong_self_reference"
    if any(ch not in vocab for ch in question + answer):
        return "out_of_vocab"
    if re.search(r"(AI|模型|系统提示|作为人工智能)", answer, re.I):
        return "meta"
    if re.search(r"(像|如同|仿佛|宛如|好似|好比|杀了你|弄死你|去死)", answer):
        return "style_or_violence"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=r"E:\espllm_data\persona_corpus\raw_alignment_distill_1024.jsonl")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.batch_size <= 16:
        raise SystemExit("count must be positive and batch-size must be 1..16")

    load_env()
    base, key = os.environ.get("QN_BASE", ""), os.environ.get("QN_KEY", "")
    model = os.environ.get("QN_MODEL", "")
    if not base or not key or not model:
        raise SystemExit("QN_BASE, QN_KEY, and QN_MODEL must be set in api.env")
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    existing_files = [
        Path(r"E:\espllm_data\persona_corpus\raw_stance26.jsonl"),
        Path(r"E:\espllm_data\persona_corpus\raw_failure_multiturn_stance26.jsonl"),
        Path(r"E:\espllm_data\persona_corpus\raw_distill_deepseek_flash_1024.jsonl"),
        Path(r"E:\espllm_data\persona_corpus\raw_alignment_distill_pilot16.jsonl"),
    ]
    seen_q, seen_a = load_existing(existing_files)
    rows = []
    if partial.exists():
        with partial.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("u") and row.get("a"):
                    rows.append(row)
                    seen_q.add(norm(row["u"]))
                    seen_a.add(norm(row["a"]))
        print(f"resuming {len(rows)} accepted rows from checkpoint", flush=True)
    vocab = set(json.loads(VOCAB_PATH.read_text(encoding="utf-8"))["chars"])
    banned_chars = set()
    rng = random.Random(20261024)
    rejected, usage = collections.Counter(), collections.Counter()
    requests, seconds = 0, 0.0
    max_requests = ((args.count + args.batch_size - 1) // args.batch_size) * 12

    while len(rows) < args.count and requests < max_requests:
        cards = []
        for _ in range(min(args.batch_size, args.count - len(rows))):
            topic = rng.choice(TOPICS)
            task = rng.choice(TASKS)
            cards.append({"topic": topic, "task": task})
        requests += 1
        try:
            result = None
            for attempt in range(3):
                try:
                    result, elapsed, used = api_request(
                        add_oov_guidance(build_prompt(cards), banned_chars),
                        base, key, model, args.timeout
                    )
                    break
                except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"network failed after retries: {exc}") from exc
                    time.sleep(3 * (attempt + 1))
            seconds += elapsed
            usage.update({k: int(v) for k, v in used.items() if isinstance(v, int)})
            generated = result.get("rows") if isinstance(result, dict) else None
            if not isinstance(generated, list):
                rejected["bad_shape"] += len(cards)
                continue
            for raw, card in zip(generated, cards):
                error = validate(raw, card, seen_q, seen_a, vocab)
                if error:
                    rejected[error] += 1
                    if error == "out_of_vocab":
                        banned_chars.update(ch for ch in str(raw.get("u", "")) + str(raw.get("a", ""))
                                            if ch not in vocab)
                    continue
                row = {
                    "src": model, "t": card["topic"], "kind": str(raw.get("kind", "")),
                    "u": str(raw["u"]).strip(), "a": str(raw["a"]).strip(),
                }
                rows.append(row)
                seen_q.add(norm(row["u"]))
                seen_a.add(norm(row["a"]))
                if len(rows) >= args.count:
                    break
            partial.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("w", encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            rejected[type(exc).__name__] += len(cards)
            print(f"request {requests} failed: {exc}", flush=True)
        if requests % 4 == 0 or len(rows) >= args.count:
            print(f"accepted={len(rows)}/{args.count} requests={requests}", flush=True)

    if len(rows) < args.count:
        raise SystemExit(f"Only {len(rows)} valid rows after {requests} requests; checkpoint: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)

    print(json.dumps({
        "output": str(output), "teacher": model, "rows": len(rows), "requests": requests,
        "elapsed_sec": round(seconds, 2), "usage": dict(usage),
        "rejected": dict(rejected),
        "topics": dict(collections.Counter(row["t"] for row in rows)),
        "task_kinds": dict(collections.Counter(row["kind"] for row in rows)),
        "unique_questions": len({norm(row["u"]) for row in rows}),
        "unique_answers": len({norm(row["a"]) for row in rows}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

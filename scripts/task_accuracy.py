"""Downstream task accuracy — the reliable precision metric for this model.

WHY NOT LOGPROB (see docs/SM120_MOE_REPORT.md §4.1):
  mean prompt-logprob is NOT a usable cross-config precision metric on M3.
  MoE top-4 routing is discrete, so a one-bit float difference flips an
  expert set and moves a token's logprob by several nats.  Measured: 27% of
  positions move by >1.0 nat, max 7.2 nats, while the mean moves only 0.05.
  A paired test over the archived native/marlin run gives t=0.698,
  95% CI [-0.102, +0.214] -> not significant.
  Within one instance logprob IS deterministic (6 repeats: spread 0.0000),
  so keep using it for same-instance debugging only.

This harness scores discrete task accuracy instead, over a fixed,
auto-graded item set (GSM8K-style arithmetic, MMLU-style MC, code, zh/en QA).

Usage:
  python3 /data/task_accuracy.py <tag> [--reps 2] [--concurrency 4] [--key KEY]
Output: /data/taskacc_<tag>.json
"""
import argparse
import concurrent.futures as cf
import json
import re
import statistics as st
import urllib.request

BASE = "http://127.0.0.1:8000"
DEFAULT_KEY = "REPLACE_WITH_YOUR_API_KEY"

SUFFIX = " End your reply with 'ANSWER: <answer>' on the last line."

# ---------------------------------------------------------------- items
# (id, category, prompt, grader)
MATH = [
    ("m01", "The store has 47 apples and sells 18, then receives 25 more. How many apples now?",
     lambda t: "54" in t, "54"),
    ("m02", "A train travels 240 km in 3 hours, then 180 km in 2 hours. What is its average speed for the whole trip in km/h?",
     lambda t: re.search(r"\b84\b", t) is not None, "84"),
    ("m03", "What is 173 times 29 minus 451?",
     lambda t: "4566" in t.replace(",", ""), "4566"),
    ("m04", "A rectangle has perimeter 56 cm and length 17 cm. What is its width in cm?",
     lambda t: re.search(r"\b11\b", t) is not None, "11"),
    ("m05", "What is the sum of the prime numbers between 10 and 20?",
     lambda t: "70" in t, "70"),
    ("m06", "What is the 20th Fibonacci number (with F1=1, F2=1)?",
     lambda t: "6765" in t.replace(",", ""), "6765"),
    ("m07", "If 3x + 7 = 31, what is the value of x?",
     lambda t: re.search(r"\b8\b", t) is not None, "8"),
    ("m08", "What is 15 percent of 240?",
     lambda t: "36" in t, "36"),
    ("m09", "A factory produces 1250 units per hour. How many units in 8 hours?",
     lambda t: "10000" in t.replace(",", ""), "10000"),
    ("m10", "What is 7 factorial (7!)?",
     lambda t: "5040" in t.replace(",", ""), "5040"),
    ("m11", "A book costs 45 dollars after a 25 percent discount. What was the original price in dollars?",
     lambda t: "60" in t, "60"),
    ("m12", "What is 128 divided by 8, plus 19?",
     lambda t: re.search(r"\b35\b", t) is not None, "35"),
    ("m13", "Three workers finish a job in 12 days. Working at the same rate, how many days do 6 workers need?",
     lambda t: re.search(r"\b6\b", t) is not None, "6"),
    ("m14", "What is the area in square cm of a circle with radius 5? Use pi = 3.14.",
     lambda t: "78.5" in t, "78.5"),
    ("m15", "A ladder 13 m long leans against a wall with its base 5 m from the wall. How high up the wall does it reach, in metres?",
     lambda t: re.search(r"\b12\b", t) is not None, "12"),
    ("m16", "What is 2 to the power of 12?",
     lambda t: "4096" in t.replace(",", ""), "4096"),
]

MC = [
    ("c01", "Which of these is NOT a prime number? A) 2  B) 9  C) 11  D) 13",
     lambda t: re.search(r"\bB\b|9", t) is not None, "B"),
    ("c02", "What does ACID stand for in database transactions?",
     lambda t: "Atomicity" in t and "Durability" in t, "Atomicity, Consistency, Isolation, Durability"),
    ("c03", "What is the time complexity of binary search on a sorted array of n elements? A) O(n)  B) O(log n)  C) O(n log n)  D) O(1)",
     lambda t: re.search(r"\bB\b|log\s*\(?n", t) is not None, "B"),
    ("c04", "Which data structure uses FIFO ordering? A) Stack  B) Queue  C) Tree  D) Graph",
     lambda t: re.search(r"\bB\b|[Qq]ueue", t) is not None, "B"),
    ("c05", "Which protocol is stateless by design? A) HTTP  B) FTP  C) SSH  D) Telnet",
     lambda t: re.search(r"\bA\b|HTTP", t) is not None, "A"),
    ("c06", "What is the derivative of x squared? A) x  B) 2x  C) x squared over 2  D) 2",
     lambda t: re.search(r"\bB\b|2x", t) is not None, "B"),
    ("c07", "In a paged KV cache, what does the block table map? A) tokens to embeddings  B) logical page indices to physical pages  C) layers to heads  D) queries to keys",
     lambda t: re.search(r"\bB\b|physical", t) is not None, "B"),
    ("c08", "Which of these is a lossless compression algorithm? A) JPEG  B) MP3  C) Huffman coding  D) Opus",
     lambda t: re.search(r"\bC\b|Huffman", t) is not None, "C"),
    ("c09", "What is the output of the Python expression 2 ** 3 ** 2?",
     lambda t: "512" in t, "512"),
    ("c10", "What is the output of list(range(10))[2:8:2]?",
     lambda t: "[2, 4, 6]" in t or "[2,4,6]" in t, "[2, 4, 6]"),
    ("c11", "In FP8 E4M3 format, approximately what is the largest finite magnitude?",
     lambda t: "448" in t, "448"),
    ("c12", "Which component decides which experts process a token in a sparse MoE layer? A) the router/gating network  B) layer norm  C) rotary embedding  D) the softmax of the final layer",
     lambda t: re.search(r"\bA\b|router|gat", t, re.I) is not None, "A"),
]

CODE = [
    ("k01", "Write a Python function count_vowels(s) that returns the number of vowels in the string s.",
     lambda t: "def count_vowels" in t and ("aeiou" in t or "AEIOU" in t), ""),
    ("k02", "Write a Python function reverse_words(s) that reverses the order of whitespace-separated words in s and returns the result.",
     lambda t: "def reverse_words" in t and ("split" in t or "join" in t), ""),
    ("k03", "Write a Python function fizzbuzz(n) that returns a list from 1 to n where multiples of 3 become 'Fizz', of 5 become 'Buzz', of both become 'FizzBuzz'.",
     lambda t: "def fizzbuzz" in t and "FizzBuzz" in t, ""),
    ("k04", "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters.",
     lambda t: "def is_palindrome" in t and ("isalnum" in t or "alnum" in t or re.search(r"[a-z\]\[]*\[", t)), ""),
    ("k05", "Write a Python function top_k(d, k) returning the k keys of dict d with the largest values, as a list.",
     lambda t: "def top_k" in t and ("sorted" in t or "heapq" in t), ""),
]

ZH = [
    ("z01", "什么是快速排序的平均时间复杂度？请简述其基本思想。",
     lambda t: ("O(n log n)" in t or "Onlogn" in t or "n log n" in t) and len(t) > 20, ""),
    ("z02", "请用一句话说明 HTTP 与 HTTPS 的主要区别。",
     lambda t: ("加密" in t or "TLS" in t or "SSL" in t) and len(t) > 10, ""),
    ("z03", "一个水池有进水管和出水管，单开进水管 6 小时注满，单开出水管 8 小时放空。两管齐开，几小时注满？",
     lambda t: re.search(r"24\s*/\s*7|3\s*\.\s*4[23]|3又3/7", t) is not None, "24/7 小时"),
    ("z04", "请简述什么是数据库索引，以及为什么它能加速查询。",
     lambda t: len(t) > 30 and ("索引" in t or "index" in t.lower()), ""),
    ("z05", "把这句话翻译成英文：「分页缓存能显著降低长上下文推理的显存占用。」",
     lambda t: ("paged" in t.lower() or "cache" in t.lower()) and len(t) > 20, ""),
    ("z06", "《红楼梦》的作者是谁？",
     lambda t: "曹雪芹" in t, "曹雪芹"),
    ("z07", "光在真空中的传播速度约为多少千米每秒？",
     lambda t: "30" in t and ("万" in t or "300000" in t or "3e5" in t.lower()), ""),
]

EN = [
    ("e01", "In one sentence, what does the GIL do in CPython?",
     lambda t: len(t) > 20 and ("thread" in t.lower() or "lock" in t.lower() or "mutex" in t.lower()), ""),
    ("e02", "What is the capital of Australia?",
     lambda t: "Canberra" in t, "Canberra"),
    ("e03", "Roughly how many kilometres is Earth's mean radius?",
     lambda t: re.search(r"6[34]\d\d", t) is not None, "~6371 km"),
    ("e04", "Explain in a sentence or two why NVFP4 quantization keeps a per-block scale factor.",
     lambda t: len(t) > 50 and "scale" in t.lower(), ""),
    ("e05", "Which chemical element has the symbol Fe?",
     lambda t: "iron" in t.lower() or "Fe" in t, "Iron"),
]

INSTR = [
    ("i01", "List exactly three primary colours as a JSON array and nothing else.",
     lambda t: t.strip().startswith("[") and t.strip().rstrip(".").endswith("]"), ""),
    ("i02", "Reply with the single word BANANA in uppercase letters only, with no other text or punctuation.",
     lambda t: t.strip().strip(".").strip() == "BANANA", ""),
    ("i03", "Output the integers from 1 to 10 separated by commas, with no other text.",
     lambda t: t.strip().startswith("1") and "10" in t and "\n" not in t.strip(), ""),
]

ITEMS = ([("math", i) for i in MATH] + [("mc", i) for i in MC]
         + [("code", i) for i in CODE] + [("zh", i) for i in ZH]
         + [("en", i) for i in EN] + [("instr", i) for i in INSTR])


def post(key, path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run_item(key, cat, item, max_tokens):
    iid, prompt, grader, gold = item
    o = post(key, "/v1/chat/completions", {
        "model": "MiniMax-M3.1-Flash",
        "messages": [{"role": "user", "content": prompt + SUFFIX}],
        "temperature": 0.0, "max_tokens": max_tokens})
    msg = o["choices"][0]["message"]
    text = msg.get("content") or ""
    rsn = msg.get("reasoning") or msg.get("reasoning_content") or ""
    try:
        ok = bool(grader(text))
    except Exception:
        ok = False
    return {"id": iid, "cat": cat, "pass": ok, "gold": gold,
            "len": len(text), "reasoning_len": len(rsn), "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--out", default="/data")
    a = ap.parse_args()

    jobs = [(cat, it, r) for r in range(a.reps) for cat, it in ITEMS]
    results = []
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(run_item, a.key, cat, it, a.max_tokens): (cat, it, r)
                for cat, it, r in jobs}
        for f in cf.as_completed(futs):
            cat, it, r = futs[f]
            rec = f.result()
            rec["rep"] = r
            results.append(rec)
            mark = "PASS" if rec["pass"] else "FAIL"
            print(f"  rep{r} {rec['id']:5s} {mark}  len={rec['len']:5d}", flush=True)

    # per-category and per-item pass rate across reps
    percat, peritem = {}, {}
    for rec in results:
        percat.setdefault(rec["cat"], []).append(rec["pass"])
        peritem.setdefault(rec["id"], []).append(rec["pass"])
    cat_summary = {c: (sum(v), len(v)) for c, v in percat.items()}
    # strict = passed in every rep; loose = passed in at least one rep
    strict = sum(1 for v in peritem.values() if all(v))
    loose = sum(1 for v in peritem.values() if any(v))
    nitems = len(peritem)
    overall = sum(1 for r in results if r["pass"])

    print(f"\n[{a.tag}] OVERALL {overall}/{len(results)} request-passes")
    for c in ("math", "mc", "code", "zh", "en", "instr"):
        if c in cat_summary:
            p, t = cat_summary[c]
            print(f"  {c:6s} {p:3d}/{t:3d}  ({100.0*p/t:.1f}%)")
    print(f"  strict (pass in all {a.reps} reps) = {strict}/{nitems}"
          f"   loose (pass in >=1 rep) = {loose}/{nitems}")
    fails = [r["id"] for r in results if not r["pass"]]
    if fails:
        print("  failing:", " ".join(sorted(set(fails))))

    json.dump({"tag": a.tag, "reps": a.reps, "results": results,
               "cat_summary": cat_summary, "strict": strict, "loose": loose,
               "nitems": nitems, "request_passes": overall,
               "nrequests": len(results)},
              open(f"{a.out}/taskacc_{a.tag}.json", "w"), ensure_ascii=False)
    print("SAVED", f"{a.out}/taskacc_{a.tag}.json")


if __name__ == "__main__":
    main()

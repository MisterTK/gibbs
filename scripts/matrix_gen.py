#!/usr/bin/env python3
"""Generate bench/matrix.json: every model variant <= 10 GB that gibbs should run on Metal.

Set A: OpenRouter models with a Hugging Face id from a frontier-lab org, with their unsloth
       conversions (priority: OpenRouter `created`, newest first).
Set B: every other unsloth repo whose name contains a known model-family keyword
       (priority: HF `lastModified`, newest first).
Variants: every .gguf (shards grouped, mmproj/MTP/imatrix dropped) and one safetensors variant per
repo, kept when the total size is <= 10,000,000,000 bytes.  Responses are cached under
target/matrix/cache so re-runs only fetch what changed; existing bench/matrix.json state is
unaffected (matrix_run.py keys state by item id).
"""
import json, os, re, subprocess, sys, time, urllib.error, urllib.request

LIMIT = 10_000_000_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "target", "matrix", "cache")
os.makedirs(CACHE, exist_ok=True)
LABS = {"google", "Qwen", "meta-llama", "mistralai", "microsoft", "deepseek-ai", "openai", "nvidia",
        "ibm-granite", "LiquidAI", "zai-org", "HuggingFaceTB", "allenai", "moonshotai", "MiniMaxAI",
        "XiaomiMiMo", "tencent", "stepfun-ai", "CohereLabs", "inclusionAI", "arcee-ai", "baidu",
        "ByteDance-Seed"}
FAMILY = ("gemma qwen llama mistral ministral magistral devstral phi- deepseek gpt-oss nemotron granite "
          "lfm glm smollm olmo kimi minimax mimo hunyuan hy- ernie command aya ling trinity step seed").split()
SUFFIXES = ["", "-GGUF", "-bnb-4bit", "-unsloth-bnb-4bit", "-FP8", "-FP8-Dynamic", "-NVFP4"]
UPSTREAM_SUFFIXES = ["-GGUF", "-FP8", "-NVFP4"]
TAGS = {"text-generation", "image-text-to-text", "any-to-any", None}
EXCLUDE = re.compile(r"embed|rerank|tts|asr|whisper", re.I)
TOKEN = os.environ.get("HF_TOKEN")


def get(url, raw=False):
    """GET with exponential backoff on 429/5xx. Returns (body, headers); 404 -> (None, {})."""
    delay = 2.0
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"User-Agent": "gibbs-matrix-gen"})
        if TOKEN and "huggingface.co" in url:
            req.add_header("Authorization", f"Bearer {TOKEN}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return (body if raw else json.loads(body)), r.headers
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, {}
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2


def cached(key, fn):
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", key) + ".json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 3 * 86400:
        return json.load(open(path))
    v = fn()
    json.dump(v, open(path, "w"))
    return v


def list_unsloth():
    out, url = [], "https://huggingface.co/api/models?author=unsloth&limit=1000&full=true"
    while url:
        body, hdr = get(url)
        out += body
        m = re.search(r'<([^>]+)>;\s*rel="next"', hdr.get("Link", "") or "")
        url = m.group(1) if m else None
    return out


def tree(repo):
    def fetch():
        files, url = [], f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
        while url:
            body, hdr = get(url)
            if body is None:
                return None
            files += [{"path": f["path"], "size": (f.get("lfs") or {}).get("size", f.get("size", 0))}
                      for f in body if f.get("type") == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', hdr.get("Link", "") or "")
            url = m.group(1) if m else None
        return files
    return cached("tree_" + repo, fetch)


def quant_of_config(repo):
    def fetch():
        body, _ = get(f"https://huggingface.co/{repo}/resolve/main/config.json", raw=True)
        if body is None:
            return {}
        try:
            c = json.loads(body)
        except ValueError:
            return {}
        c = c.get("text_config", c) if "quantization_config" not in c else c
        q = c.get("quantization_config") or {}
        return {"quant": q.get("quant_method") or q.get("quant_type"), "dtype": c.get("torch_dtype") or c.get("dtype")}
    return cached("config_" + repo, fetch)


SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def variants(repo, family, created):
    files = tree(repo)
    if not files:
        return []
    items, shards = [], {}
    for f in files:
        p = f["path"]
        low = p.lower()
        if p.endswith(".gguf"):
            if "mmproj" in low or "mtp/" in low or "mtp-" in low or "imatrix" in low:
                continue
            m = SHARD.match(p)
            if m:
                s = shards.setdefault(m.group(1), {"first": None, "bytes": 0})
                s["bytes"] += f["size"]
                if m.group(2) == "00001":
                    s["first"] = p
            else:
                items.append((p, f["size"]))
    items += [(s["first"], s["bytes"]) for s in shards.values() if s["first"]]
    out = []
    for p, size in items:
        if size <= LIMIT:
            fmt = re.sub(r"\.gguf$", "", os.path.basename(p))
            fmt = SHARD.sub(r"\1", fmt + ".gguf") if SHARD.match(fmt + ".gguf") else fmt
            tag = re.search(r"((?:UD-)?(?:IQ|Q|TQ|MXFP|NVFP|BF|F)\d[A-Za-z0-9_]*)$", fmt)
            out.append({"id": f"{repo}/{p}", "repo": repo, "file": p, "bytes": size,
                        "format": tag.group(1) if tag else fmt, "family": family,
                        "openrouter_created": created})
    st = sum(f["size"] for f in files if f["path"].endswith(".safetensors"))
    if 0 < st <= LIMIT:
        c = quant_of_config(repo)
        out.append({"id": repo, "repo": repo, "file": None, "bytes": st,
                    "format": "safetensors-" + str(c.get("quant") or c.get("dtype") or "unknown"),
                    "family": family, "openrouter_created": created})
    return out


def family_of(name):
    low = name.lower()
    return next((k.strip("-") for k in FAMILY if k in low), "other")


def main():
    body, _ = get("https://openrouter.ai/api/v1/models")
    orm = [m for m in body["data"] if m.get("hugging_face_id") and m["hugging_face_id"].split("/")[0] in LABS]
    orm.sort(key=lambda m: -(m.get("created") or 0))
    unsloth = {m["id"].lower(): m for m in cached("unsloth_list", list_unsloth)}
    print(f"openrouter: {len(orm)} lab models; unsloth: {len(unsloth)} repos", file=sys.stderr)

    ok_tag = lambda m: m.get("pipeline_tag") in TAGS and not EXCLUDE.search(m["id"])
    order, seen_repo = [], set()  # (repo, family, created)
    for m in orm:
        org, base = m["hugging_face_id"].split("/", 1)
        created = m.get("created")
        cands = [f"unsloth/{base}{s}" for s in SUFFIXES] + [f"{org}/{base}"] + [f"{org}/{base}{s}" for s in UPSTREAM_SUFFIXES]
        for c in cands:
            if c.lower() in seen_repo or EXCLUDE.search(c):
                continue
            info = unsloth.get(c.lower())
            if info is None and not c.lower().startswith("unsloth/"):
                info = cached("info_" + c, lambda c=c: get(f"https://huggingface.co/api/models/{c}")[0] or {}) or None
            if info and ok_tag(info):
                seen_repo.add(c.lower())
                order.append((info["id"], family_of(base), created))
    rest = sorted((m for k, m in unsloth.items() if k not in seen_repo and ok_tag(m)
                   and any(w in k for w in FAMILY)), key=lambda m: m.get("lastModified", ""), reverse=True)
    order += [(m["id"], family_of(m["id"]), None) for m in rest]
    print(f"repos to scan: {len(order)}", file=sys.stderr)

    items, ids = [], set()
    for i, (repo, fam, created) in enumerate(order):
        try:
            vs = variants(repo, fam, created)
        except Exception as e:  # network hiccup: keep going; a later run fills it in
            print(f"skip {repo}: {e}", file=sys.stderr)
            continue
        for v in vs:
            if v["id"] not in ids:
                ids.add(v["id"])
                items.append(v)
        if i % 50 == 0:
            print(f"  {i}/{len(order)} repos, {len(items)} items", file=sys.stderr)
    out = {"generated": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(),
           "items": items}
    json.dump(out, open(os.path.join(ROOT, "bench", "matrix.json"), "w"), indent=0)
    print(f"wrote bench/matrix.json: {len(items)} items", file=sys.stderr)


if __name__ == "__main__":
    main()

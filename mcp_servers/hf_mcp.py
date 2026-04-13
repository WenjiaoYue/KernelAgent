#!/usr/bin/env python3
"""
Minimal MCP server for HuggingFace Hub queries.
Protocol: MCP 2024-11-05 over stdio JSON-RPC 2.0

Tools exposed:
  - list_recent_models(days, limit, sort, pipeline_tag)
  - get_model_info(model_id)
"""
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

# Honor proxy env vars (https_proxy / http_proxy)
_proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or ""
_proxy_handler = urllib.request.ProxyHandler(
    {"https": _proxy, "http": _proxy} if _proxy else {}
)
_opener = urllib.request.build_opener(_proxy_handler)

# HuggingFace API token (avoids 429 rate limits on anonymous requests)
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

# ─── Whitelist / filter configuration ────────────────────────────────────
TARGET_ORGS = {
    "Qwen", "moonshotai", "deepseek-ai", "THUDM", "tencent",
    "meituan-longcat", "zai-org", "XiaomiMiMo", "MiniMaxAI",
    "miromind-ai", "openai", "mistralai", "Wan-AI", "microsoft",
    "stepfun-ai", "google", "ByteDance-Seed",
}

EXCLUDED_KEYWORDS = {"gguf", "fp8", "fp4", "int4", "int8", "base", "bf16"}

PIPELINE_TAGS = ["text-generation", "image-text-to-text"]

# Cache for OSI-approved license IDs
_osi_cache: set | None = None

def _get_osi_licenses() -> set:
    """Fetch and cache OSI-approved SPDX license IDs (lowercase)."""
    global _osi_cache
    if _osi_cache is not None:
        return _osi_cache
    try:
        url = "https://spdx.org/licenses/licenses.json"
        headers = {"User-Agent": "mcp-server-hf/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with _opener.open(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        _osi_cache = {
            item["licenseId"].lower()
            for item in data.get("licenses", [])
            if item.get("isOsiApproved")
        }
    except Exception:
        _osi_cache = set()
    return _osi_cache


def _check_osi_license(model_id: str) -> bool:
    """Check if a model has an OSI-approved license."""
    osi = _get_osi_licenses()
    if not osi:
        return True  # If we can't fetch the list, don't block
    try:
        info = hf_api(f"models/{model_id}")
        # Check tags for license:xxx
        for tag in info.get("tags", []):
            if tag.startswith("license:"):
                lid = tag.split("license:", 1)[1].lower()
                return lid in osi
        # Check cardData
        card = info.get("cardData") or {}
        if "license_name" in card:
            return card["license_name"].lower() in osi
        if "license" in card:
            lid = card["license"]
            if isinstance(lid, str):
                return lid.lower() in osi
    except Exception:
        pass
    return False


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def hf_api(path, params=None):
    import time
    url = "https://huggingface.co/api/" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "mcp-server-hf/1.0"}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with _opener.open(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("HuggingFace API request failed after retries")


def list_recent_models(days=15, limit=10, sort="trending_score", pipeline_tag=""):
    """List recent models filtered by org whitelist, excluded keywords, and OSI license."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Which pipeline tags to search
    tags_to_search = [pipeline_tag] if pipeline_tag else PIPELINE_TAGS

    # Gather candidates from all pipeline tags
    all_candidates = []
    seen_ids = set()
    for tag in tags_to_search:
        params = {
            "sort": sort,
            "direction": -1,
            "limit": 50,
            "full": "false",
        }
        if tag:
            params["pipeline_tag"] = tag
        try:
            models = hf_api("models", params)
        except Exception:
            continue
        for m in models:
            mid = m.get("id", "")
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_candidates.append(m)

    results = []
    skipped = {"org": 0, "keyword": 0, "date": 0, "license": 0}

    for m in all_candidates:
        if len(results) >= limit:
            break

        mid = m.get("id", "")

        # 1) Org whitelist
        if "/" not in mid:
            skipped["org"] += 1
            continue
        org = mid.split("/")[0]
        if org not in TARGET_ORGS:
            skipped["org"] += 1
            continue

        # 2) Excluded keywords
        mid_lower = mid.lower()
        if any(kw in mid_lower for kw in EXCLUDED_KEYWORDS):
            skipped["keyword"] += 1
            continue

        # 3) Date filter
        created = m.get("createdAt", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if dt < cutoff:
                    skipped["date"] += 1
                    continue
            except ValueError:
                pass

        # 4) OSI license check
        if not _check_osi_license(mid):
            skipped["license"] += 1
            continue

        results.append({
            "id": mid,
            "pipeline_tag": m.get("pipeline_tag", ""),
            "likes": m.get("likes", 0),
            "downloads": m.get("downloads", 0),
            "createdAt": created,
        })

    return {
        "models": results,
        "filters_applied": {
            "target_orgs": sorted(TARGET_ORGS),
            "excluded_keywords": sorted(EXCLUDED_KEYWORDS),
            "days": days,
            "osi_license_required": True,
        },
        "skipped": skipped,
    }


def get_model_info(model_id):
    return hf_api(f"models/{model_id}")


TOOLS = [
    {
        "name": "list_recent_models",
        "description": (
            "List recent HuggingFace models filtered by: org whitelist "
            "(Qwen, deepseek-ai, openai, mistralai, google, microsoft, etc.), "
            "excluded keywords (gguf, fp8, int4, base, etc.), "
            "pipeline tags (text-generation, image-text-to-text), "
            "and OSI-approved license. Returns models matching all criteria."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (default 15)",
                    "default": 15,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                    "default": 10,
                },
                "sort": {
                    "type": "string",
                    "description": "Sort field: trending_score (default) or createdAt or downloads or likes",
                    "default": "trending_score",
                },
                "pipeline_tag": {
                    "type": "string",
                    "description": "Filter by specific task type. If empty, searches text-generation and image-text-to-text",
                    "default": "",
                },
            },
        },
    },
    {
        "name": "get_model_info",
        "description": "Get detailed metadata for a specific HuggingFace model by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID, e.g. 'meta-llama/Llama-3-8B'",
                },
            },
            "required": ["model_id"],
        },
    },
]


def handle(req):
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-server-hf", "version": "1.0.0"},
            },
        })

    elif method in ("notifications/initialized", "initialized"):
        pass  # notification, no response

    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if tool_name == "list_recent_models":
                result = list_recent_models(
                    days=int(args.get("days", 10)),
                    limit=int(args.get("limit", 20)),
                    sort=args.get("sort", "createdAt"),
                    pipeline_tag=args.get("pipeline_tag", ""),
                )
                text = json.dumps(result, indent=2, ensure_ascii=False)
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                })
            elif tool_name == "get_model_info":
                result = get_model_info(args.get("model_id", ""))
                text = json.dumps(result, indent=2, ensure_ascii=False)
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                })
            else:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                })
        except Exception as e:
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                },
            })

    elif req_id is not None:
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle(req)


if __name__ == "__main__":
    main()

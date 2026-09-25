"""
실제 기업 이용약관 → article_v2 인코더 판정 → 조항 지목률 측정

사용법
------
    # URL 직접:
    .venv/bin/python3 -m backend.eval.real_tos_eval --url https://example.com/terms --name "회사명"

    # 로컬 텍스트 파일:
    .venv/bin/python3 -m backend.eval.real_tos_eval --file /path/to/tos.txt --name "회사명"

    # 여러 건 일괄 (파일 형식: 줄당  URL<TAB>이름):
    .venv/bin/python3 -m backend.eval.real_tos_eval --batch backend/eval/tos_targets.tsv

GPT 호출 없음 — article_v2(KoELECTRA) 인코더만 사용해 비용 0.
결과는 data/eval/real_tos/ 아래 JSON으로 저장된다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시)
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.agents.judgment_agent import predict_articles

# analyze.py는 fastapi를 상단 임포트하므로 로직만 복사
_MIN_CLAUSE_CHARS = 20
_MAX_CLAUSES = 60


def split_clauses(text: str) -> tuple[list[str], int]:
    import os
    max_c = int(os.environ.get("MAX_CLAUSES", str(_MAX_CLAUSES)))
    parts = re.split(
        r"(?=제\s*\d+\s*조|^\s*\d+\.\s|^[①②③④⑤⑥⑦⑧⑨⑩]|\n{2,})",
        text,
        flags=re.MULTILINE,
    )
    found = [s.strip() for s in parts if len(s.strip()) > _MIN_CLAUSE_CHARS]
    return found[:max_c], max(0, len(found) - max_c)

_OUT_DIR = _ROOT / "data" / "eval" / "real_tos"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# ── HTML → 순수 텍스트 ──────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """script/style/nav/footer를 버리고 본문 텍스트만 수집한다.

    블록 태그(p, h1~h6, li, div, section, article) 경계에 \\n\\n을 삽입해
    split_clauses()의 \\n{2,} 분리자가 작동하도록 한다.
    네이버처럼 제N조 형식이 없는 헤더 문체 약관에도 단락 분리가 된다.
    """

    _SKIP  = {"script", "style", "nav", "footer", "header", "noscript", "iframe"}
    _BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div",
              "section", "article", "td", "th", "br"}

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK and not self._skip_depth:
            self._buf.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK and not self._skip_depth:
            self._buf.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._buf.append(stripped)

    def get_text(self) -> str:
        return "".join(self._buf)


def _fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL 오류: {e.reason}") from e

    parser = _TextExtractor()
    parser.feed(html)
    text = parser.get_text()

    # JS SPA 감지 — 의미있는 조항이 없으면 Playwright로 재시도
    clauses, _ = split_clauses(text)
    if len(clauses) < 3:
        print("  → JS 렌더링 감지, Playwright로 재시도...")
        text = _fetch_with_playwright(url)

    return text


def _fetch_with_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9"},
            user_agent=_HEADERS["User-Agent"],
        )
        try:
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle", timeout=15000)
            return page.inner_text("body")
        except Exception as e:
            raise RuntimeError(f"Playwright 실패: {e}") from e
        finally:
            browser.close()


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


# ── 핵심 평가 ──────────────────────────────────────────────────────────────


def _run_one(text: str, name: str, source: str) -> dict:
    clauses, truncated = split_clauses(text)
    total = len(clauses)

    results: list[dict] = []
    for clause in clauses:
        articles = predict_articles(clause)
        results.append({
            "text": clause[:300],   # 저장 크기 절약
            "articles": articles,
            "flagged": bool(articles),
        })

    flagged_n = sum(1 for r in results if r["flagged"])
    flagged_rate = flagged_n / total if total else 0.0

    # 조별 지목 횟수
    article_counts: dict[str, int] = {}
    for r in results:
        for art in r["articles"]:
            article_counts[art] = article_counts.get(art, 0) + 1

    return {
        "name": name,
        "source": source,
        "total_clauses": total,
        "truncated": truncated,
        "flagged": flagged_n,
        "flag_rate": round(flagged_rate * 100, 1),
        "article_counts": dict(sorted(article_counts.items(), key=lambda x: -x[1])),
        "clauses": results,
    }


def _save(result: dict) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w가-힣]", "_", result["name"])[:30]
    path = _OUT_DIR / f"{slug}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_summary(r: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {r['name']}")
    print(f"  소스: {r['source']}")
    print(f"{'='*55}")
    print(f"  전체 조항: {r['total_clauses']}  (잘림: {r['truncated']})")
    print(f"  지목된 조항: {r['flagged']}  ({r['flag_rate']}%)")
    if r["article_counts"]:
        print(f"  조별 지목 횟수:")
        for art, cnt in r["article_counts"].items():
            print(f"    {art}: {cnt}회")
    print()
    print("  [지목된 조항 미리보기]")
    shown = 0
    for c in r["clauses"]:
        if c["flagged"] and shown < 5:
            arts = " · ".join(c["articles"])
            preview = c["text"].replace("\n", " ")[:120]
            print(f"  [{arts}]  {preview}…")
            shown += 1
    if shown == 0:
        print("  없음")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="실제 약관 → 인코더 판정")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--url",   help="단일 URL")
    grp.add_argument("--file",  help="로컬 텍스트 파일 경로")
    grp.add_argument("--batch", help="TSV 파일 (URL<TAB>이름)")
    ap.add_argument("--name", default="unknown", help="회사/서비스 이름 (--url, --file 시)")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []  # (text_or_url, name)

    if args.batch:
        for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                targets.append((parts[0].strip(), parts[1].strip()))
            else:
                targets.append((parts[0].strip(), parts[0].strip()))
    elif args.url:
        targets.append((args.url, args.name))
    else:
        targets.append((args.file, args.name))

    all_results = []
    for source, name in targets:
        print(f"처리 중: {name}  ({source[:60]})")
        try:
            if source.startswith("http"):
                text = _fetch_url(source)
            else:
                text = _read_file(source)
        except Exception as e:
            print(f"  ⚠ 스킵: {e}")
            continue

        result = _run_one(text, name, source)
        path = _save(result)
        _print_summary(result)
        print(f"  → {path}")
        all_results.append(result)

    if len(all_results) > 1:
        print("\n[전체 요약]")
        total_c = sum(r["total_clauses"] for r in all_results)
        total_f = sum(r["flagged"] for r in all_results)
        print(f"  총 {len(all_results)}개 서비스  조항 {total_c}개  지목 {total_f}개 "
              f"({total_f/total_c*100:.1f}%)")


if __name__ == "__main__":
    main()

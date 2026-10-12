# backend/scripts/backup.py
"""다시 만들 수 없는 데이터·모델·DB 사용자 테이블을 암호화된 파일 하나로 묶는다.

`data/`·`models/`는 git에 없어서 디스크가 죽으면 같이 사라진다. 결과 파일은 서버 밖으로
옮겨야 백업이 된다. DB `accounts`에 OAuth 토큰이 있어 암호화 없이는 만들지 않는다.

실행(시작할 때 암호를 두 번 묻는다 — 잃어버리면 복구 불가):
    .venv/bin/python -m backend.scripts.backup
    .venv/bin/python -m backend.scripts.backup --dry-run      # 담길 목록과 크기만

복구:
    gpg -d <파일>.tar.zst.gpg | zstd -d | tar -xf - -C <복구할 폴더>
"""

import argparse
import datetime
import getpass
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.utils import PROJECT_ROOT

BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", str(PROJECT_ROOT / "backups")))

INCLUDE = [
    "data",
    "models/README.md",
    "models/article_v1",
    "models/article_v2",   # 가중치 삭제됨(2026-10-12) — metrics·임계값만 남음
    "models/article_v3",
    *sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "models").glob("sllm_lora_*")),
]
# 다시 만들 수 있고 크기만 큰 것(LightRAG 인덱스 231MB, 결론 난 실험)
EXCLUDE = [
    "data/eval/archive/lightrag_full_storage",
    "data/eval/archive/lightrag_pilot_storage",
    "__pycache__",
]
DB_TABLES = ["users", "accounts", "sessions", "verification_token", "analyses", "alembic_version"]


def _du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        rel = p.relative_to(PROJECT_ROOT)
        if any(rel.as_posix().startswith(e + "/") or e in rel.parts for e in EXCLUDE):
            continue
        if p.is_file():
            total += p.stat().st_size
    return total


def dump_db(out_dir: Path) -> None:
    import psycopg2

    from backend.db.connection import DATABASE_URL

    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    conn.set_session(readonly=True)
    try:
        with conn.cursor() as cur:
            for t in DB_TABLES:
                with open(out_dir / f"{t}.csv", "w", encoding="utf-8", newline="") as f:
                    cur.copy_expert(f'COPY (SELECT * FROM "{t}") TO STDOUT WITH CSV HEADER', f)
    finally:
        conn.close()


def ask_passphrase() -> str:
    print("\n백업 암호를 정합니다. 입력한 글자는 화면에 안 보입니다. **영문 입력 상태**에서 치세요.")
    for _ in range(3):
        p1 = getpass.getpass("  암호(8자 이상): ")
        if len(p1) < 8:
            print("  8자보다 짧습니다 — 다시")
            continue
        if getpass.getpass("  한 번 더: ") == p1:
            return p1
        print("  두 번 입력이 다릅니다 — 다시")
    sys.exit("3번 실패해서 멈췄습니다. 아무것도 만들지 않았습니다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = [p for p in INCLUDE if not (PROJECT_ROOT / p).exists()]
    if missing:
        sys.exit(f"없는 경로: {missing}")

    total = 0
    for p in INCLUDE:
        size = _du(PROJECT_ROOT / p)
        total += size
        print(f"  {size / 1e9:6.2f} GB  {p}")
    print(f"  {total / 1e9:6.2f} GB  합계(압축 전) + DB 테이블 {len(DB_TABLES)}개")
    if args.dry_run:
        return

    passphrase = ask_passphrase()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = BACKUP_DIR / f"contractrisk_{stamp}.tar.zst.gpg"

    with tempfile.TemporaryDirectory() as tmp:
        db_dir = Path(tmp) / "db_dump"
        db_dir.mkdir()
        dump_db(db_dir)
        print(f"  DB 테이블 {len(DB_TABLES)}개 내보냄")

        print("  압축·암호화 중 — 몇 분 걸립니다")
        tar = subprocess.Popen(
            ["tar", "-C", str(PROJECT_ROOT), *[f"--exclude={e}" for e in EXCLUDE], "-cf", "-", *INCLUDE,
             "-C", tmp, "db_dump"],
            stdout=subprocess.PIPE,
        )
        zst = subprocess.Popen(["zstd", "-T0", "-3", "-q"], stdin=tar.stdout, stdout=subprocess.PIPE)
        tar.stdout.close()
        # pinentry 창 대신 파이프로 암호를 넘긴다 — 명령행 인자로 넘기면 `ps`에 보인다
        r, w = os.pipe()
        os.write(w, passphrase.encode() + b"\n")
        os.close(w)
        gpg = subprocess.run(
            ["gpg", "--batch", "--yes", "--pinentry-mode", "loopback", "--passphrase-fd", str(r),
             "--no-symkey-cache", "--symmetric", "--cipher-algo", "AES256", "--output", str(out)],
            stdin=zst.stdout, pass_fds=(r,),
        )
        os.close(r)
        zst.stdout.close()
        if tar.wait() != 0 or zst.wait() != 0 or gpg.returncode != 0:
            out.unlink(missing_ok=True)
            sys.exit("실패 — 불완전한 파일은 지웠다")

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    Path(f"{out}.sha256").write_text(f"{h.hexdigest()}  {out.name}\n")
    print(f"\n완료: {out} ({out.stat().st_size / 1e9:.2f} GB)")
    print(f"sha256: {h.hexdigest()}")


if __name__ == "__main__":
    main()

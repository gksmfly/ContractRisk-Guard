# backend/preprocess/cleaner.py
import re

_SENT_SPLIT   = re.compile(r"(?<=[.?!。])\s+|(?<=\n)\s*")
_CIRCLE_NUMS  = {chr(0x2460 + i): f"({i + 1})" for i in range(20)}  # ①-⑳
_PARTY_INFO   = re.compile(
    r"【(?:원고|피고|상고인|피상고인|원심판결|항소인|피항소인|신청인|피신청인|채권자|채무자|참가인|선정당사자)[^】]*】[^【\n]*"
)


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("　", " ").replace("\xa0", " ")
    for circle, replacement in _CIRCLE_NUMS.items():
        text = text.replace(circle, replacement)
    text = text.replace("「", "").replace("」", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_precedent_content(text: str) -> str:
    """판례내용 전용 — 당사자/법원 메타정보 제거 후 clean_text 적용."""
    text = _PARTY_INFO.sub("", text)
    return clean_text(text)


def split_chunks(text: str, chunk_size: int, overlap: int, min_chunk: int) -> list[str]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(sent) > chunk_size:
            if current and len(current) >= min_chunk:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sent), chunk_size - overlap):
                piece = sent[i:i + chunk_size]
                if len(piece) >= min_chunk:
                    chunks.append(piece)
            continue

        if len(current) + len(sent) + 1 > chunk_size:
            if len(current) >= min_chunk:
                chunks.append(current.strip())
            current = current[-overlap:] + " " + sent if overlap and current else sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current and len(current) >= min_chunk:
        chunks.append(current.strip())
    return chunks

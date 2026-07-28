# backend/api/auth.py
"""요청마다 OpenAI 호출 + GPU 임베딩이 도는 분석 엔드포인트를 보호하는
최소한의 공유 비밀키 인증.

프론트엔드(Next.js API route, 서버 사이드)만 이 키를 알고 있고 브라우저에는
절대 노출되지 않으므로, 별도 사용자 계정 시스템 없이 "백엔드는 우리 프론트만
호출한다"는 요구를 충족하는 가장 단순한 방식이다. 더 세분화된 인가(사용자별
권한 등)가 필요해지면 이 dependency를 JWT/OAuth로 교체한다.
"""

import hmac
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not API_KEY:
        # 로컬 개발 편의를 위해 API_KEY가 설정되지 않았으면 인증을 건너뛴다.
        # 배포 환경에서는 반드시 .env에 API_KEY를 설정해야 한다.
        return
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="인증되지 않은 요청입니다.")

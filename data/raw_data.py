import requests
import json
import os

def save_raw_law_data(auth_key, mst_id, filename):
    """
    국가법령 API를 호출하여 원본 JSON을 파일로 저장합니다.
    MST: 법령일련번호 (약관법: 253527 등)
    """
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": auth_key,        # 서연님의 인증키 
        "target": "law",       # 법령 대상 
        "MST": mst_id,         # 법령일련번호 [cite: 114, 129]
        "type": "JSON"         # 응답 형식 [cite: 110]
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        raw_data = response.json()
        
        # 'data/raw' 폴더에 저장 (아이디어 초안의 원본 데이터 저장 방식 반영) 
        os.makedirs('data/raw', exist_ok=True)
        with open(f'data/raw/{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=4)
        
        print(f"성공: {filename}.json 원본 데이터 저장 완료")
        return raw_data
    else:
        print(f"API 호출 실패: {response.status_code}")
        return None

# 실행 예시 (약관법 예시 ID: 253527)
# my_key = "서연님의_인증키"
# save_raw_law_data(my_key, "253527", "raw_act_on_terms")

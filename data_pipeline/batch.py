import json
import os
import requests

def extract_and_save_data(data_path: str, output_path: str):
    """
    [모듈화] 
    Go Extractor Service(포트 8080)와 통신하여 
    대용량 JSON 파일에서 AI-Human 대화 쌍을 추출하고 결과를 저장합니다.
    (parser.py의 압축 로직과 완전히 분리되었습니다.)
    """
    abs_path = os.path.abspath(data_path)
    
    print(f"데이터 추출 요청 중 (Target: {abs_path})...")
    try:
        response = requests.post(
            "http://localhost:8080/extract",
            json={"file_path": abs_path},
            timeout=60 # 대용량 파일 파싱을 위해 넉넉한 타임아웃 지정
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Go Extractor Service와의 통신 중 에러 발생: {e}")
        print("Go 서버가 8080 포트에서 실행 중인지 확인해주세요.")
        return

    print(f"Go 서버 파싱 완료! 총 {len(data)}건의 대화 쌍을 가져왔습니다.")
    
    # 결과를 JSON 파일로 저장 (추후 parser.py가 이 파일을 읽어서 처리하도록 모듈화)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"파싱된 원시 결과물이 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 추출 테스트: final_data.json에서 데이터를 뽑아 extracted_pairs.json에 저장
    extract_and_save_data("../final_data.json", "extracted_pairs.json")

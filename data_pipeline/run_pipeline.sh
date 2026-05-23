#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

echo "=== 데이터 파이프라인 시작 ==="

# 1. Go Extractor Service 백그라운드 실행
echo "[1/3] Go Extractor Service 시작 중..."
cd extractor_service
go run main.go &
GO_PID=$!
cd ..

# Go 서버가 기동할 시간을 잠깐 줍니다 (2초)
sleep 2

# 2. batch.py 실행 (데이터 추출)
echo "[2/3] batch.py 실행 (데이터 추출 중)..."
# 에러 발생해도 Go 프로세스는 죽이고 스크립트를 종료하도록 트랩 설정
trap "kill $GO_PID" EXIT
python3 batch.py

# batch.py 처리가 끝나면 Go 서버 종료
echo "Go Extractor Service 종료 중..."
kill $GO_PID
# 정상 종료이므로 트랩 해제
trap - EXIT

# 3. refine_batch.py 실행 (AI 모델로 정제 및 분석)
echo "[3/3] refine_batch.py 실행 (AI 모델 정제 중)..."
python3 refine_batch.py

echo "=== 모든 데이터 파이프라인 처리가 완료되었습니다! ==="
echo "결과물: data_pipeline/final_seeds.json"

package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/buger/jsonparser"
)

type ExtractedPair struct {
	PrevUserText string `json:"prev_user_text"`
	AIText       string `json:"ai_text"`
	UserText     string `json:"user_text"`
	UUID         string `json:"uuid"`
}

type ExtractRequest struct {
	FilePath string `json:"file_path"`
}

func extractHandler(w http.ResponseWriter, r *http.Request) {
	var req ExtractRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	data, err := os.ReadFile(req.FilePath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var pairs []ExtractedPair

	// 오직 final_data.json 전용 하드코딩 추출
	jsonparser.ArrayEach(data, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		uuid, _ := jsonparser.GetString(value, "conv_id")
		if uuid == "" {
			uuid = "unknown_conv_id"
		}

		var lastUserText, currentAIText, lastRole string

		jsonparser.ArrayEach(value, func(msg []byte, msgType jsonparser.ValueType, off int, err error) {
			role, _ := jsonparser.GetString(msg, "role")
			text, _ := jsonparser.GetString(msg, "content")

			if role == "user" {
				// User(t-1) -> AI(t) -> User(t) 삼원조 추출
				if lastRole == "assistant" && currentAIText != "" && lastUserText != "" {
					pairs = append(pairs, ExtractedPair{
						PrevUserText: lastUserText,
						AIText:       currentAIText,
						UserText:     text,
						UUID:         uuid,
					})
				}
				// 현재 User의 텍스트를 다음 사이클의 원인이 되는 prev_user_text로 업데이트
				lastUserText = text
			} else if role == "assistant" {
				// AI 텍스트 업데이트
				currentAIText = text
			}
			lastRole = role
		}, "prompt")
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(pairs)
}

func main() {
	http.HandleFunc("/extract", extractHandler)
	log.Println("Go Extractor Service (final_data ONLY - Triad Mode) listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

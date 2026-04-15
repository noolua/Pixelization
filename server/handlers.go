package main

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"net/http"
	"path/filepath"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

// RegisterHandlers binds all HTTP routes to the mux.
func RegisterHandlers(mux *http.ServeMux, engine *InferenceEngine, staticDir string) {
	// GET / — serve index.html
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(staticDir, "index.html"))
	})

	// POST /pixelize
	mux.HandleFunc("/pixelize", handlePixelize(engine))

	// GET /health
	mux.HandleFunc("/health", handleHealth(engine))

	// Static files: /static/*
	fs := http.FileServer(http.Dir(staticDir))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))
}

func handlePixelize(engine *InferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse multipart form (max 32MB)
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "failed to parse form"})
			return
		}

		// Get image file
		file, _, err := r.FormFile("image")
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "missing image field"})
			return
		}
		defer file.Close()

		// Decode image
		img, _, err := image.Decode(file)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "invalid image file"})
			return
		}

		// Get cell_size (default 4, range 2-8)
		cellSize := 4
		if v := r.FormValue("cell_size"); v != "" {
			cellSize, err = strconv.Atoi(v)
			if err != nil || cellSize < 2 || cellSize > 8 {
				writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "cell_size must be between 2 and 8"})
				return
			}
		}

		// Get original_size
		originalSize := r.FormValue("original_size") == "true"

		// Preprocess
		pre, err := PreprocessImage(img, PreprocessConfig{CellSize: cellSize})
		if err != nil {
			log.Printf("Preprocess error: %v", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "preprocessing failed"})
			return
		}

		// Inference
		inputShape := ort.NewShape(1, 3, int64(pre.Height), int64(pre.Width))
		outputData, err := engine.Run(pre.Data, inputShape)
		if err != nil {
			log.Printf("Inference error: %v", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "inference failed"})
			return
		}

		// Postprocess
		pngBytes, err := Postprocess(outputData, PostprocessConfig{
			CellSize:     cellSize,
			OriginalSize: originalSize,
			Width:        pre.Width,
			Height:       pre.Height,
		})
		if err != nil {
			log.Printf("Postprocess error: %v", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "postprocessing failed"})
			return
		}

		w.Header().Set("Content-Type", "image/png")
		w.Write(pngBytes)
	}
}

func handleHealth(engine *InferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":       "ok",
			"model_loaded": engine.IsReady(),
		})
	}
}

// corsMiddleware adds CORS headers allowing all origins.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "*")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

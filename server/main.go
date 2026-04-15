package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	addr := flag.String("addr", ":8000", "listen address")
	modelPath := flag.String("model", "../tests/onnx/pixelization.onnx", "path to ONNX model")
	staticDir := flag.String("static", "../static", "path to static files directory")
	flag.Parse()

	// Set ONNX Runtime shared library path
	libPath := os.Getenv("ONNX_RUNTIME_LIB")
	if libPath == "" {
		log.Fatal("ONNX_RUNTIME_LIB environment variable is required (path to libonnxruntime.dylib/.so)")
	}
	ort.SetSharedLibraryPath(libPath)

	// Initialize ONNX environment
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to init ONNX environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Create inference engine
	engine, err := NewInferenceEngine(*modelPath)
	if err != nil {
		log.Fatalf("Failed to create inference engine: %v", err)
	}
	defer engine.Destroy()

	// Set up HTTP routes
	mux := http.NewServeMux()
	RegisterHandlers(mux, engine, *staticDir)

	srv := &http.Server{
		Addr:         *addr,
		Handler:      corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second, // long timeout for inference
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	}()

	log.Printf("Starting server on http://localhost%s", *addr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}

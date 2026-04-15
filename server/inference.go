package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// InferenceEngine wraps an ONNX Runtime session with thread-safe inference.
type InferenceEngine struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

// NewInferenceEngine loads the ONNX model and configures the best available
// execution provider (CoreML on macOS ARM, CUDA on Linux, CPU fallback).
func NewInferenceEngine(modelPath string) (*InferenceEngine, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()

	configureExecutionProvider(opts)

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"image"},
		[]string{"output"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	log.Println("ONNX session created successfully")
	return &InferenceEngine{session: session}, nil
}

// Run performs inference on the given NCHW float32 input data.
// Input shape must be (1, 3, H, W) where H and W are multiples of 4.
// Returns output data with the same shape. Thread-safe.
func (e *InferenceEngine) Run(inputData []float32, inputShape ort.Shape) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Pre-allocate output tensor with same shape (output shape == input shape for this model)
	outputSize := inputShape.FlattenedSize()
	outputData := make([]float32, outputSize)
	outputTensor, err := ort.NewTensor(inputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	if err := e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return nil, fmt.Errorf("session run: %w", err)
	}

	return outputData, nil
}

// IsReady returns true if the session is loaded.
func (e *InferenceEngine) IsReady() bool {
	return e.session != nil
}

// Destroy releases the ONNX session.
func (e *InferenceEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
}

// configureExecutionProvider tries to enable hardware acceleration.
// Falls back to CPU silently.
func configureExecutionProvider(opts *ort.SessionOptions) {
	goos := runtime.GOOS
	arch := runtime.GOARCH

	if goos == "darwin" && arch == "arm64" {
		err := opts.AppendExecutionProviderCoreML(0)
		if err == nil {
			log.Println("Using CoreML execution provider")
			return
		}
		log.Printf("CoreML not available: %v", err)
	}

	if goos == "linux" {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			defer cudaOpts.Destroy()
			err = opts.AppendExecutionProviderCUDA(cudaOpts)
			if err == nil {
				log.Println("Using CUDA execution provider")
				return
			}
		}
		log.Printf("CUDA not available: %v", err)
	}

	log.Println("Using CPU execution provider")
}

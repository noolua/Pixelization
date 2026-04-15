package main

import (
	"bytes"
	"image"
	"image/png"

	"github.com/disintegration/imaging"
)

// PostprocessConfig controls postprocessing behavior.
type PostprocessConfig struct {
	CellSize     int  // 2-8
	OriginalSize bool // if true, skip enlarge step
	Width        int  // model input/output width
	Height       int  // model input/output height
}

// Postprocess converts ONNX output tensor to PNG bytes.
// Pipeline: denormalize NCHW → shrink by /4 NEAREST → optionally enlarge by cellSize NEAREST → PNG.
func Postprocess(outputData []float32, cfg PostprocessConfig) ([]byte, error) {
	// Step 1: Denormalize NCHW [-1,1] → *image.NRGBA [0,255]
	img := fromNCHW(outputData, cfg.Width, cfg.Height)

	// Step 2: Shrink by factor of 4 (best_cell_size) with NEAREST
	shrinkW := cfg.Width / 4
	shrinkH := cfg.Height / 4
	img = imaging.Resize(img, shrinkW, shrinkH, imaging.NearestNeighbor)

	// Step 3: Conditionally enlarge by cellSize
	if !cfg.OriginalSize {
		finalW := shrinkW * cfg.CellSize
		finalH := shrinkH * cfg.CellSize
		img = imaging.Resize(img, finalW, finalH, imaging.NearestNeighbor)
	}

	// Step 4: Encode as PNG
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// fromNCHW converts NCHW float32 tensor (1,3,H,W) values [-1,1] to *image.NRGBA.
func fromNCHW(data []float32, w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	hw := h * w
	pix := img.Pix
	stride := img.Stride

	for y := 0; y < h; y++ {
		rowOffset := y * stride
		pixelRow := y * w
		for x := 0; x < w; x++ {
			pi := pixelRow + x
			idx := rowOffset + x*4

			pix[idx] = clampUint8((data[pi] + 1.0) * 127.5)      // R
			pix[idx+1] = clampUint8((data[hw+pi] + 1.0) * 127.5)  // G
			pix[idx+2] = clampUint8((data[2*hw+pi] + 1.0) * 127.5) // B
			pix[idx+3] = 255                                        // A
		}
	}

	return img
}

// clampUint8 clamps a float32 to [0, 255] and rounds to nearest uint8.
func clampUint8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

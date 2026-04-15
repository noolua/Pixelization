package main

import (
	"image"
	"math"

	"github.com/disintegration/imaging"
)

// PreprocessConfig holds preprocessing parameters.
type PreprocessConfig struct {
	CellSize int // 2-8
}

// PreprocessResult holds the preprocessed tensor data and dimensions.
type PreprocessResult struct {
	Data   []float32 // NCHW float32, length = 1 * 3 * H * W
	Width  int       // multiple of 4
	Height int       // multiple of 4
}

// PreprocessImage converts an image to NCHW float32 tensor data normalized to [-1, 1].
// Pipeline: RGB convert → rescale → resize to cell-size dims → center crop to mult of 4 → normalize.
func PreprocessImage(img image.Image, cfg PreprocessConfig) (*PreprocessResult, error) {
	// Step 1: Convert to NRGBA (RGB)
	rgba := imaging.Clone(img)

	// Step 2: Rescale (>4000px halve, <128px double)
	rgba = rescale(rgba)

	// Step 3: Resize to ((w/cellSize)*4, (h/cellSize)*4) BICUBIC
	w := rgba.Bounds().Dx()
	h := rgba.Bounds().Dy()
	targetW := (w / cfg.CellSize) * 4
	targetH := (h / cfg.CellSize) * 4
	rgba = imaging.Resize(rgba, targetW, targetH, imaging.CatmullRom)

	// Step 4: Center crop to multiples of 4
	rgba = centerCropToMultipleOf4(rgba)

	// Step 5: Normalize to NCHW [-1, 1]
	result := toNCHW(rgba)
	return result, nil
}

// rescale ensures image dimensions are between 128 and 4000 pixels.
// Replicates test_pro.py:rescale().
func rescale(img *image.NRGBA) *image.NRGBA {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()

	for w > 4000 || h > 4000 {
		w /= 2
		h /= 2
		img = imaging.Resize(img, w, h, imaging.CatmullRom)
	}

	for w < 128 || h < 128 {
		w *= 2
		h *= 2
		img = imaging.Resize(img, w, h, imaging.CatmullRom)
	}

	return img
}

// centerCropToMultipleOf4 crops the image from center so dimensions are multiples of 4.
// Replicates test_pro.py:process() crop logic.
func centerCropToMultipleOf4(img *image.NRGBA) *image.NRGBA {
	ow := img.Bounds().Dx()
	oh := img.Bounds().Dy()

	nw := int(math.Round(float64(ow)/4.0)) * 4
	nh := int(math.Round(float64(oh)/4.0)) * 4

	if nw > ow {
		nw = ow
	}
	if nh > oh {
		nh = oh
	}

	left := (ow - nw) / 2
	top := (oh - nh) / 2

	return imaging.Crop(img, image.Rect(left, top, left+nw, top+nh))
}

// toNCHW converts *image.NRGBA to NCHW float32 normalized to [-1, 1].
// Output layout: [R channel plane, G channel plane, B channel plane].
func toNCHW(img *image.NRGBA) *PreprocessResult {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	hw := h * w
	data := make([]float32, 3*hw)

	stride := img.Stride
	pix := img.Pix

	for y := 0; y < h; y++ {
		rowOffset := y * stride
		pixelRow := y * w
		for x := 0; x < w; x++ {
			idx := rowOffset + x*4
			pi := pixelRow + x

			// (pixel/255 - 0.5) / 0.5 == pixel/127.5 - 1.0
			data[0*hw+pi] = float32(pix[idx]) / 127.5 - 1.0   // R
			data[1*hw+pi] = float32(pix[idx+1]) / 127.5 - 1.0  // G
			data[2*hw+pi] = float32(pix[idx+2]) / 127.5 - 1.0  // B
		}
	}

	return &PreprocessResult{
		Data:   data,
		Width:  w,
		Height: h,
	}
}

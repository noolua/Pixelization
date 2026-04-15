package main

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"math/rand"
	"sort"
)

// KMeansOptimize reduces the number of colors in an NRGBA image using
// weighted K-Means clustering on unique colors (frequency-weighted).
// Returns PNG-encoded bytes.
func KMeansOptimize(img *image.NRGBA, targetColors int) ([]byte, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	pix := img.Pix
	stride := img.Stride

	// Step 1: collect unique colors and their frequencies
	type colorFreq struct {
		r, g, b uint8
		count   int
	}
	colorMap := make(map[uint32]*colorFreq)
	for y := 0; y < h; y++ {
		row := y * stride
		for x := 0; x < w; x++ {
			idx := row + x*4
			packed := uint32(pix[idx])<<16 | uint32(pix[idx+1])<<8 | uint32(pix[idx+2])
			if cf, ok := colorMap[packed]; ok {
				cf.count++
			} else {
				colorMap[packed] = &colorFreq{pix[idx], pix[idx + 1], pix[idx + 2], 1}
			}
		}
	}

	uniqueCount := len(colorMap)
	if uniqueCount <= targetColors {
		// Already within target, return as-is
		var buf bytes.Buffer
		png.Encode(&buf, img)
		return buf.Bytes(), nil
	}

	// Build flat arrays for efficient K-Means
	// IMPORTANT: use a sorted key list so iteration order is deterministic.
	packedKeys := make([]uint32, 0, uniqueCount)
	for k := range colorMap {
		packedKeys = append(packedKeys, k)
	}
	sortUint32(packedKeys)

	colors := make([][3]float64, uniqueCount)
	counts := make([]float64, uniqueCount)
	totalCount := 0.0
	for i, packed := range packedKeys {
		cf := colorMap[packed]
		colors[i] = [3]float64{float64(cf.r), float64(cf.g), float64(cf.b)}
		counts[i] = float64(cf.count)
		totalCount += counts[i]
	}

	// Step 2: initialize centers using weighted random selection
	rng := rand.New(rand.NewSource(42))
	probs := make([]float64, uniqueCount)
	for i := range counts {
		probs[i] = counts[i] / totalCount
	}

	centers := make([][3]float64, targetColors)
	chosen := make(map[int]bool)
	for k := 0; k < targetColors; k++ {
		// weighted random selection without replacement
		idx := weightedRandom(rng, probs, chosen)
		chosen[idx] = true
		centers[k] = colors[idx]
	}

	// Step 3: iterate K-Means (max 20 rounds)
	labels := make([]int, uniqueCount)
	for iter := 0; iter < 20; iter++ {
		// Assign each unique color to nearest center
		for i, c := range colors {
			bestDist := math.MaxFloat64
			bestK := 0
			for k, center := range centers {
				dr := c[0] - center[0]
				dg := c[1] - center[1]
				db := c[2] - center[2]
				d := dr*dr + dg*dg + db*db
				if d < bestDist {
					bestDist = d
					bestK = k
				}
			}
			labels[i] = bestK
		}

		// Update centers (weighted average)
		newCenters := make([][3]float64, targetColors)
		clusterWeights := make([]float64, targetColors)
		for i, c := range colors {
			k := labels[i]
			w := counts[i]
			newCenters[k][0] += c[0] * w
			newCenters[k][1] += c[1] * w
			newCenters[k][2] += c[2] * w
			clusterWeights[k] += w
		}

		converged := true
		for k := range newCenters {
			if clusterWeights[k] > 0 {
				inv := 1.0 / clusterWeights[k]
				nc := [3]float64{
					newCenters[k][0] * inv,
					newCenters[k][1] * inv,
					newCenters[k][2] * inv,
				}
				dr := nc[0] - centers[k][0]
				dg := nc[1] - centers[k][1]
				db := nc[2] - centers[k][2]
				if dr*dr+dg*dg+db*db > 1.0 {
					converged = false
				}
				centers[k] = nc
			}
		}
		if converged {
			break
		}
	}

	// Step 4: round centers to uint8
	centerRGB := make([][3]uint8, targetColors)
	for k, c := range centers {
		centerRGB[k] = [3]uint8{
			clampFloat(c[0]),
			clampFloat(c[1]),
			clampFloat(c[2]),
		}
	}

	// Step 5: build mapping from packed color -> new packed color
	// Uses the same sorted packedKeys to ensure index alignment with labels[].
	colorMapNew := make(map[uint32]uint32, uniqueCount)
	for i, packed := range packedKeys {
		rgb := centerRGB[labels[i]]
		colorMapNew[packed] = uint32(rgb[0])<<16 | uint32(rgb[1])<<8 | uint32(rgb[2])
	}

	// Step 6: apply mapping to produce new image
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	outPix := out.Pix
	copy(outPix, pix)

	for y := 0; y < h; y++ {
		row := y * stride
		for x := 0; x < w; x++ {
			idx := row + x*4
			packed := uint32(pix[idx])<<16 | uint32(pix[idx+1])<<8 | uint32(pix[idx+2])
			newPacked := colorMapNew[packed]
			outPix[idx] = uint8(newPacked >> 16)
			outPix[idx+1] = uint8(newPacked >> 8)
			outPix[idx+2] = uint8(newPacked)
			// alpha stays
		}
	}

	var buf bytes.Buffer
	png.Encode(&buf, out)
	return buf.Bytes(), nil
}

func weightedRandom(rng *rand.Rand, probs []float64, chosen map[int]bool) int {
	// Build cumulative distribution excluding already-chosen indices
	var total float64
	for i, p := range probs {
		if !chosen[i] {
			total += p
		}
	}
	r := rng.Float64() * total
	var cum float64
	for i, p := range probs {
		if chosen[i] {
			continue
		}
		cum += p
		if r <= cum {
			return i
		}
	}
	// fallback: return first unchosen
	for i := range probs {
		if !chosen[i] {
			return i
		}
	}
	return 0
}

func clampFloat(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

func sortUint32(a []uint32) {
	sort.Slice(a, func(i, j int) bool { return a[i] < a[j] })
}

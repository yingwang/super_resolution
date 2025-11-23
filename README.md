# Video Call Enhancement POC

A proof-of-concept web application demonstrating **receiver-side video enhancement** for WebRTC video calls. The app enhances incoming video streams in real-time using TensorFlow.js and WebGL shaders.

## Use Case

In video calling applications, the video quality received can be degraded due to:
- Network bandwidth limitations
- Compression artifacts
- Low-light conditions on the sender's side
- Low resolution cameras

This POC demonstrates how the **receiver** can enhance the incoming video locally to improve visual quality without requiring any changes on the sender's side.

## Features

- **Real-time Enhancement** - Process incoming video frames at 30+ FPS
- **Multiple Enhancement Modes**:
  - **Auto Enhance** - Automatic brightness/contrast optimization
  - **Sharpen** - Edge enhancement for clearer details
  - **Denoise** - Reduces compression artifacts and noise
  - **Super Resolution** - 2x upscaling with detail enhancement
  - **HDR Effect** - Improved dynamic range

- **Adjustable Parameters**
  - Enhancement intensity
  - Brightness adjustment
  - Contrast adjustment

- **Simulation Options** (for testing)
  - Webcam or sample video sources
  - Quality degradation simulation (720p → 240p)
  - Side-by-side comparison view

## Quick Start

```bash
# No npm required! Just use Python's built-in server:
python3 -m http.server 3000
```

Then open http://localhost:3000 in Chrome.

## How It Works

### Architecture

```
┌─────────────────┐     WebRTC      ┌─────────────────┐
│     Sender      │ ──────────────► │    Receiver     │
│  (Low quality)  │   Compressed    │                 │
└─────────────────┘     Stream      │  ┌───────────┐  │
                                    │  │ Enhancer  │  │
                                    │  │ (WebGL +  │  │
                                    │  │  TF.js)   │  │
                                    │  └─────┬─────┘  │
                                    │        ▼        │
                                    │  Enhanced Video │
                                    └─────────────────┘
```

### Processing Pipeline

1. **Receive Frame** - Video frame from WebRTC stream
2. **Upload to GPU** - Frame data sent to WebGL texture
3. **Apply Enhancement** - GPU shader processes the frame
4. **Display** - Enhanced frame rendered to canvas

### Enhancement Techniques

| Mode | Technique | Use Case |
|------|-----------|----------|
| Sharpen | Unsharp mask convolution | Blurry video |
| Denoise | Bilateral filter | Compression artifacts |
| Super Resolution | Bicubic + Laplacian sharpening | Low resolution |
| HDR | Local tone mapping | Poor lighting |
| Auto | Adaptive brightness/contrast | General improvement |

## Project Structure

```
super_resolution/
├── index.html          # Video call UI
├── styles.css          # Styling
├── js/
│   ├── enhancer.js     # WebGL shaders + TensorFlow.js
│   └── video-call.js   # Call simulation & enhancement logic
├── package.json
└── README.md
```

## Usage

1. **Start a Call**
   - Select video source (webcam or sample video)
   - Choose simulated quality level
   - Click "Start Call"

2. **Adjust Enhancement**
   - Toggle enhancement on/off
   - Select enhancement mode
   - Adjust intensity sliders

3. **Compare Results**
   - Click "Compare" to see original vs enhanced side-by-side

## Browser Support

- **Chrome 90+** (recommended)
- **Firefox 88+**
- **Edge 90+**
- Safari 15+ (limited WebGL support)

## Performance

| Resolution | Mode | Expected FPS |
|------------|------|--------------|
| 720p | Sharpen | 60 fps |
| 720p | Super-res | 30 fps |
| 480p | Any | 60 fps |
| 1080p | Sharpen | 45 fps |

*Performance varies based on GPU capabilities*

## Limitations

- **Not true AI super-resolution**: Real-time constraints prevent using deep learning models like ESRGAN. The super-resolution mode uses enhanced bicubic interpolation.
- **Single-page POC**: This simulates a video call locally. Real WebRTC peer connections would require a signaling server.
- **CORS restrictions**: Sample videos must have appropriate headers.

## Future Improvements

- [ ] Integrate actual WebRTC peer connection
- [ ] Add WebCodecs API for more efficient video processing
- [ ] Implement face-aware enhancement
- [ ] Add bandwidth-adaptive quality modes
- [ ] WebGPU support for newer browsers

## License

MIT License

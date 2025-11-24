# How to Verify ESPCN Trained Weights Are Loaded

After training your ESPCN model, follow these steps to verify the weights are being used:

## 1. Check the Model Files

After training completes, you should see:
```
espcn_tfjs/
├── model.json          # Model architecture
├── group1-shard1of1.bin  # Model weights
└── espcn_loader.js     # Helper script
```

Check the timestamp of these files:
```bash
ls -lh espcn_tfjs/
```

The files should have recent modification times matching when you finished training.

## 2. Verify in Browser Console

Open your web app and:
1. Open browser DevTools (F12)
2. Go to the Console tab
3. Select "ESPCN Super Resolution" from the enhancement dropdown

You should see one of these messages:

### ✓ Trained Weights Loaded Successfully:
```
Attempting to load pre-trained model from espcn_tfjs/model.json...
✓ Pre-trained ESPCN model loaded successfully!
  - Layers: 4
  - Parameters: 24,108
  - Input shape: [batch, height, width, 3]
  - Output scale: 2x
ESPCN model ready (pretrained: true)
```

### ⚠ Fallback to Crafted Weights:
```
Attempting to load pre-trained model from espcn_tfjs/model.json...
⚠ Pre-trained model not available: [error message]
No pre-trained weights found, using crafted weights...
ESPCN model ready (pretrained: false)
```

## 3. Check the UI Notification

When you select ESPCN, look for the notification message:

- **Trained weights**: "ESPCN ready! ✓ Trained weights (24,108 params)"
- **Crafted weights**: "ESPCN ready! ⚠ Crafted weights (24,108 params)"

## 4. Inspect Model Info

In the browser console, after ESPCN loads, you'll see:
```javascript
ESPCN Model Info: {
  name: "ESPCN",
  scale: 2,
  layers: 4,
  params: 24108,
  pretrained: true  // <-- This should be true for trained weights
}
```

## 5. Check Network Tab

1. Open DevTools Network tab
2. Reload the page
3. Select ESPCN enhancement mode
4. Look for requests to:
   - `espcn_tfjs/model.json` (should return 200 OK)
   - `espcn_tfjs/group1-shard1of1.bin` (should return 200 OK)

If these files load successfully, your trained weights are being used.

## Troubleshooting

### Model not found
- Make sure `espcn_tfjs/` folder is in the same directory as `index.html`
- Check file permissions: `chmod -R 755 espcn_tfjs/`
- Verify the web server is serving the files (not blocked by CORS)

### Wrong path
- The default path is `espcn_tfjs/model.json`
- If you moved the files, update the path in `js/espcn.js`:
  ```javascript
  this.pretrainedModelPath = 'your/path/to/model.json';
  ```

### Old weights cached
- Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Clear browser cache
- Check the file modification time matches when you trained

## Visual Quality Check

The best way to verify is visual comparison:

1. Load a compressed video (low bitrate)
2. Compare enhancement with:
   - **Trained weights**: Should show sharp details, reduced noise
   - **Crafted weights**: Basic upscaling, less effective denoising

Trained weights should perform noticeably better on compressed video with artifacts.

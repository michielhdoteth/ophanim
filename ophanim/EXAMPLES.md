# Ophanim Examples

## Basic Usage

### 1. Check System Status

```bash
ophanim status
```

Output:

```
+-- System Status --+
| GPU          | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM Free    | 5.7 GB                            |
| Safe Mode    | OFF                               |
| Models       | lmstudio:google/gemma-4-e2b        |
+-------------------+
```

### 2. Probe a Video

```bash
ophanim probe warehouse_clip.mp4
```

Output:

```
+-- Video Probe: warehouse_clip.mp4 --+
| Duration   | 30.5s (30s)             |
| Resolution | 1920x1080               |
| FPS        | 30.00                   |
| Codec      | avc1                    |
| Frames     | 915                     |
| Cost       | MEDIUM                  |
+--------------------------------------+
```

### 3. Observe a Video

```bash
ophanim observe warehouse_clip.mp4 --mode balanced
```

Output:

```
+-- Observation Summary --+
| A worker enters, inspects a pallet, picks up a box, and exits. |
+---------------------------+

Timeline:
  [00:03] Worker enters from right side of frame.
  [00:11] Worker bends near pallet, appears to inspect contents.
  [00:18] Worker picks up a cardboard box.
  [00:24] Worker exits frame carrying box.

Entities: worker, pallet, cardboard, box, frame, contents

Artifacts: runs/20260521_143022_abc123/
```

### 4. Ask a Specific Question

```bash
ophanim ask warehouse_clip.mp4 "What color is the box?"
```

Output:

```
Question: What color is the box?
Confidence: HIGH

+-- Answer --+
| The box is brown cardboard color. It appears to be a standard shipping box. |
+-------------+

Evidence:
  [00:18] The worker picks up a brown cardboard box from the pallet.
```

### 5. JSON Output for Agents

```bash
ophanim ask warehouse_clip.mp4 "Is there a gas cylinder?" --json
```

Output:

```json
{
  "answer": "Yes, a gas cylinder appears near the left wall around 00:08.",
  "evidence": [
    {
      "timestamp": "00:08",
      "time_seconds": 8.0,
      "answer": "A gas cylinder is visible against the left wall",
      "confidence": "medium"
    }
  ],
  "confidence": "high"
}
```

### 6. Segment an Object

```bash
ophanim segment warehouse_clip.mp4 "gas cylinder" --start 5 --end 15
```

Output:

```
+-- Segmentation Complete --+
| Processed 6 frames for 'gas cylinder' |
+----------------------------+

| Object ID          | Timestamps | Masks |
|--------------------|------------|-------|
| gas_cylinder_5     | 00:05      | 1     |
| gas_cylinder_8     | 00:08      | 1     |
| gas_cylinder_12    | 00:12      | 1     |

Masks saved to: runs/20260521_.../masks/
```

### 7. Track an Object

```bash
ophanim track warehouse_clip.mp4 "worker" --start 0 --end 30
```

Output:

```
+-- Track: worker_warehouse_clip --+
| The 'worker' was tracked across 12 frames from 00:03 to 00:24. |
+----------------------------------+

| Time  | BBox              | Confidence |
|-------|--------------------|------------|
| 00:03 | [120, 220, 200, 400] | 0.82      |
| 00:05 | [130, 225, 210, 405] | 0.85      |
| 00:08 | [200, 230, 280, 410] | 0.79      |
```

### 8. Process with Different Modes

```bash
# Fast mode for quick check
ophanim observe long_video.mp4 --mode fast

# Detailed mode for short high-value clips
ophanim observe short_clip.mp4 --mode detailed

# Dense search mode for finding specific objects
ophanim ask clip.mp4 "Find the red box" --mode dense
```

### 9. Save Observation as Memory

```bash
ophanim observe sales_call.mp4 --save-memory
```

This creates `memory/videos/2026-05-21-sales-call.mp4.md` with the observation.

View saved memories:

```bash
ophanim memory list
ophanim memory view --name "2026-05-21-sales-call"
```

### 10. Cache Control

```bash
# Uses cached result (fast, same params)
ophanim observe video.mp4

# Force reprocess with different params
ophanim observe video.mp4 --fps 1.0 --mode detailed --force

# Cache is invalidated when file changes
```

## Configuration

Create a custom config:

```bash
set OPHANIM_CONFIG=my_custom_config.yaml
ophanim status
```

Example custom config override:

```yaml
# my_custom_config.yaml
defaults:
  mode: fast
  max_resolution: 512
  fps: 0.25

models:
  vlm:
    base_url: "http://192.168.1.100:1234/v1"
```

## Error Recovery

When you get an error:

```bash
# GPU OOM - downgrade mode
ophanim observe video.mp4 --mode fast

# Model not found - check LM Studio
ophanim status

# Video not found - check path
ophanim probe /correct/path/to/video.mp4
```

## Pipeline Script

```bash
# Analyze a security camera clip
ophanim probe security_clip.mp4
ophanim observe security_clip.mp4 --mode fast --json > observation.json
ophanim ask security_clip.mp4 "Is there a person?" --json >> observation.json
ophanim segment security_clip.mp4 "person" --start 0 --end 60
```

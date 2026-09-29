# ComfyUI-ShortsRemake

Helper nodes for the "split a video into segments -> VLM prompts -> regenerate each segment
with a swapped person -> concatenate" pipeline. Workflows using them live in
`C:\00_forensic\ComfyUI-Easy-Install\workflow\` (see `README_Shorts_Remake.md` there).

| Node | Role |
|---|---|
| Shorts Video Segments | video file -> per-segment frame batches (list). `split_mode` fixed / scene |
| Shorts Prompts Collector | collects QwenVL JSON answers (per segment + common) -> `prompts.json`, `comfyui_prompts.txt` |
| Shorts Prompts Loader | `prompts.json` -> per-clip positive / pose prompt / skip_frames / frame_count lists + negative, size, fps |
| Shorts Clip Saver | VIDEO -> `clip_NN.mp4` next to prompts.json |
| Shorts Concat | all clip paths -> `final.mp4` (ffmpeg), also returns a VIDEO for Save Video |

Outputs marked as lists make every downstream node execute once per segment, which is how a
normal single-clip generation graph becomes a per-segment loop without loop nodes.

Requires OpenCV (bundled with ComfyUI portable) and ffmpeg (imageio-ffmpeg or PATH).

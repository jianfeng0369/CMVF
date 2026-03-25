import os
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from fusion_model import get_fusion_model
import args

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_model_path():
    return os.path.join(SCRIPT_DIR, args.model_save_path, "best_model.pth")

def _index_video_files(path):
    if os.path.isfile(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return {stem: path}

    if not os.path.isdir(path):
        raise FileNotFoundError(f"找不到视频路径: {path}")

    videos = {}
    for name in sorted(os.listdir(path)):
        full_path = os.path.join(path, name)
        if not os.path.isfile(full_path):
            continue
        if not name.lower().endswith(VIDEO_EXTENSIONS):
            continue

        stem = os.path.splitext(name)[0]
        if stem in videos:
            raise ValueError(f"检测到重复视频名 stem: {stem}")
        videos[stem] = full_path

    if not videos:
        raise FileNotFoundError(f"路径下未找到视频文件: {path}")

    return videos


def _collect_video_pairs(vi_path, ir_path):
    if os.path.isfile(vi_path) and os.path.isfile(ir_path):
        output_name = f"{os.path.splitext(os.path.basename(vi_path))[0]}.mp4"
        return [(vi_path, ir_path, output_name)]

    if os.path.isdir(vi_path) and os.path.isdir(ir_path):
        vi_videos = _index_video_files(vi_path)
        ir_videos = _index_video_files(ir_path)

        common_stems = sorted(set(vi_videos) & set(ir_videos))
        if not common_stems:
            raise FileNotFoundError("vi 与 ir 目录下没有同名视频可配对")

        missing_vi = sorted(set(ir_videos) - set(vi_videos))
        missing_ir = sorted(set(vi_videos) - set(ir_videos))
        if missing_vi or missing_ir:
            print(f"跳过未配对视频, vi缺失: {missing_vi}, ir缺失: {missing_ir}")

        return [(vi_videos[stem], ir_videos[stem], f"{stem}.mp4") for stem in common_stems]

    raise ValueError("vi_path 与 ir_path 必须同时是文件，或同时是目录")


def _prepare_ir_tensor(ir_frame, to_tensor, device):
    if ir_frame.ndim == 3:
        ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    return to_tensor(ir_frame).unsqueeze(0).to(device)


def _fuse_frame(model, vi_frame_bgr, ir_frame, to_tensor, device):
    vi_frame_rgb = cv2.cvtColor(vi_frame_bgr, cv2.COLOR_BGR2RGB)
    vi_tensor = to_tensor(vi_frame_rgb)
    ir_tensor = _prepare_ir_tensor(ir_frame, to_tensor, device)

    vi_np = vi_tensor.permute(1, 2, 0).cpu().numpy()
    vi_ycrcb = cv2.cvtColor(vi_np, cv2.COLOR_RGB2YCrCb)
    vi_y = torch.from_numpy(vi_ycrcb[:, :, 0:1]).permute(2, 0, 1).unsqueeze(0).to(device)

    ir_tensor = ir_tensor.tile(1, 2, 1, 1)
    vi_y = vi_y.tile(1, 2, 1, 1)

    fusion = model(ir_tensor, vi_y)
    fusion_y = fusion.squeeze(0).squeeze(0).cpu().numpy()
    fusion_y = np.clip(fusion_y, 0.0, 1.0)

    vi_ycrcb[:, :, 0] = fusion_y
    fusion_rgb = cv2.cvtColor(vi_ycrcb, cv2.COLOR_YCrCb2RGB)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)
    fusion_rgb = (fusion_rgb * 255.0).round().astype(np.uint8)

    return cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)


def _open_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    return cap


def _get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count


def test(test_vi_path, test_ir_path, fusion_results_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    os.makedirs(fusion_results_path, exist_ok=True)

    model = get_fusion_model().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model.eval()

    video_pairs = _collect_video_pairs(test_vi_path, test_ir_path)

    with torch.no_grad():
        for vi_video_path, ir_video_path, output_name in video_pairs:
            vi_cap = _open_video_capture(vi_video_path)
            ir_cap = _open_video_capture(ir_video_path)

            start_time = time.time()
            output_path = os.path.join(fusion_results_path, output_name)

            try:
                vi_fps, vi_width, vi_height, vi_frames = _get_video_info(vi_cap)
                _, ir_width, ir_height, ir_frames = _get_video_info(ir_cap)

                if vi_width != ir_width or vi_height != ir_height:
                    raise ValueError(
                        f"视频分辨率不一致: vi=({vi_width}, {vi_height}), ir=({ir_width}, {ir_height})"
                    )

                if vi_fps <= 0:
                    vi_fps = 25.0

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    vi_fps,
                    (vi_width, vi_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建输出视频: {output_path}")

                processed = 0
                progress_step = 50
                try:
                    while True:
                        vi_ok, vi_frame = vi_cap.read()
                        ir_ok, ir_frame = ir_cap.read()

                        if vi_ok != ir_ok:
                            raise RuntimeError("vi 与 ir 视频帧数不一致，无法继续逐帧融合")
                        if not vi_ok:
                            break

                        fusion_frame = _fuse_frame(model, vi_frame, ir_frame, to_tensor, device)
                        writer.write(fusion_frame)
                        processed += 1
                        if processed % progress_step == 0:
                            print(f"{os.path.basename(output_name)} 已处理 {processed} 帧")
                finally:
                    writer.release()

                elapsed = time.time() - start_time
                print(
                    f"{os.path.basename(output_name)} 完成: "
                    f"{processed} 帧, vi帧数={vi_frames}, ir帧数={ir_frames}, "
                    f"fps={vi_fps:.2f}, 用时={elapsed:.2f}s"
                )
            finally:
                vi_cap.release()
                ir_cap.release()


if __name__ == "__main__":

    path_vi = f"../data/videos/vi"
    path_ir = f"../data/videos/ir"
    path_op = f"../data/videos/fusion"

    output_path = f"{path_op}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    IR_PATH = path_ir
    VI_PATH = path_vi
    SAVE_PATH = output_path

    test_vi_path = VI_PATH
    test_ir_path = IR_PATH
    fusion_results_path = SAVE_PATH

    model_path = _resolve_model_path()
    test(test_vi_path, test_ir_path, fusion_results_path, model_path)

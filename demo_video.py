import copy
import cv2
import json
import argparse
import subprocess
import ffmpeg
from typing import NamedTuple
from src import model, util
from src.body import Body


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(
        command_array,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return FFProbeResult(return_code=result.returncode, json=result.stdout, error=result.stderr)


class Writer:
    def __init__(self, output_file, fps, pix_fmt, vcodec):
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.fps = fps
        self.output_file = output_file
        self.ff_proc = None

    def __del__(self, exc_type, exc_val, exc_tb):
        if self.ff_proc is not None:
            self.ff_proc.stdin.close()
            self.ff_proc.wait()

    def __call__(self, frame):
        if self.ff_proc is None:
            ffmpeg_inp = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s="%sx%s" % (frame.shape[:2][1], frame.shape[:2][0]),
                r=self.fps,
            )
            self.ff_proc = (
                ffmpeg_inp.output(output_file, pix_fmt=self.pix_fmt, vcodec=self.vcodec)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

        self.ff_proc.stdin.write(frame.tobytes())


if __name__ == "__main__":
    body_estimation = Body("model/body_pose_model.pth")
    video_file = "/home/tom/output.mp4"

    cap = cv2.VideoCapture(video_file)

    # get video file info
    ffprobe_result = ffprobe(video_file)
    info = json.loads(ffprobe_result.json)
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    fps = videoinfo["avg_frame_rate"]
    pix_fmt = videoinfo["pix_fmt"]
    vcodec = videoinfo["codec_name"]

    # define a writer object to write to a movidified file
    postfix = info["format"]["format_name"].split(",")[0]
    output_file = ".".join(video_file.split(".")[:-1]) + ".processed." + postfix

    writer = Writer(output_file, fps, pix_fmt, vcodec)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        canvas = copy.deepcopy(frame)
        candidate, subset = body_estimation(frame)
        posed_frame = util.draw_bodypose(canvas, candidate, subset)

        writer(posed_frame)

    cap.release()

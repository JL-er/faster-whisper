from faster_whisper import WhisperModel
from opencc import OpenCC
import torch
model_size = "/home/op/large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
# segments, info = model.transcribe("000001.wav", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))


# cc = OpenCC('t2s')  # 't2s.json'是一个常见的配置文件，用于繁体到简体的转换
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     print(cc.convert(segment.text))

def whisper(mp3_path):
    segments, info = model.transcribe(mp3_path, beam_size=5)

    cc = OpenCC('t2s')  # 't2s.json'是一个常见的配置文件，用于繁体到简体的转换
    for segment in segments:
        text = cc.convert(segment.text)
    return text

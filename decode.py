from pathlib import Path

from transformers import EncodecModel
import numpy as np
from torch import Tensor
import soundfile as sf

code_file = Path("/home/palonso/reps/xai-encodec/codes/blu/blues.00090.npy")
codes = Tensor(np.load(code_file)).int()
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
audio_values = model.decode(codes, [None])[0]

audio_values = audio_values.detach().numpy().squeeze()
sf.write("Test.wav", audio_values, 24000)

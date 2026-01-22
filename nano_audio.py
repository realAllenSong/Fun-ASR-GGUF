import numpy as np

def normalizer(audio, target_value=8192.0):
    """音频归一化处理"""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean((audio * audio), dtype=np.float32), dtype=np.float32)
    audio *= (target_value / (rms + 1e-7))
    np.clip(audio, -32768.0, 32767.0, out=audio)
    return audio.astype(np.int16)

def load_audio(audio_path, sample_rate=16000, use_normalizer=True):
    """加载音频文件并转换为 16kHz PCM"""
    from pydub import AudioSegment
    
    audio = np.array(
        AudioSegment.from_file(audio_path)
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .get_array_of_samples(),
        dtype=np.int16
    )
    
    if use_normalizer:
        audio = normalizer(audio, 8192.0)
    
    return audio

from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf

# 오디오 파일을 WAV로 변환하는 함수
def convert_to_wav(input_file, output_file="converted_audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)  # 16kHz로 변환, 모노 채널
    audio.export(output_file, format="wav")  # WAV로 내보내기
    return output_file

# 모델 및 프로세서 로드
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# CPU만 사용하도록 설정
device = "cpu"  # 무조건 CPU 사용
model = model.to(device)

def transcribe_audio(file_path):
    # 오디오 파일 읽기
    audio_input, sample_rate = sf.read(file_path)
    # 오디오 전처리
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to(device)
    # 모델 예측
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    # 텍스트 변환
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# 테스트 파일 경로 (여기에 변환할 파일 경로를 넣으면 돼)
audio_file_path = "/Users/dgsw8th66/Desktop/전공 공부 폴더/STT/새로운 녹음 11.m4a"  # 변환할 오디오 파일 경로를 지정하세요

# 음성 파일을 WAV로 변환
converted_file = convert_to_wav(audio_file_path)

# 텍스트로 변환
transcription = transcribe_audio(converted_file)
print("Transcription:", transcription)


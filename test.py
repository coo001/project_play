import whisper
import sounddevice as sd
import numpy as np
import rtmidi
from difflib import SequenceMatcher

# Whisper 모델 (정확도 높이기 위해 small 사용)
print("📦 Whisper 모델 로딩 중 (small)...")
model = whisper.load_model("small")

# 마이크 설정
samplerate = 16000
block_duration = 1  # 초 단위
blocksize = int(samplerate * block_duration)
channels = 1

# 트리거 키워드 목록
CUE_KEYWORDS = {
    "여긴 왜 이렇게 어두워": 1,
    "지금 비춰줘": 2,
    "무대 끝이야": 3
}

# MIDI cue 실행
def trigger_ma2_cue(cue_number):
    midi_out = rtmidi.MidiOut()
    ports = midi_out.get_ports()
    if not ports:
        print("❌ MIDI 포트를 찾을 수 없습니다.")
        return

    # MIDI 포트 열기 (loopMIDI 등으로 grandMA2 연결)
    midi_out.open_port(0)

    # MSC 메시지 (cue 번호만 바꾸면 됨)
    msg = [0xF0, 0x7F, 111, 0x02, 0x01, 0x01, cue_number, 0xF7]
    midi_out.send_message(msg)
    print(f"🎯 cue {cue_number} 실행됨!")

    midi_out.close_port()

# VAD (Voice Activity Detection): 에너지 기준으로 음성 판단
def is_voice_present(audio, threshold=0.01):
    volume = np.sqrt(np.mean(audio**2))
    return volume > threshold

# 유사도 비교
def is_similar(a, b, threshold=0.75):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# 음성 블록 처리
def process_audio(audio):
    audio = whisper.pad_or_trim(audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="ko", fp16=False)
    result = whisper.decode(model, mel, options)
    text = result.text.strip()
    if text:
        print(f"🎙️ 인식된 대사: {text}")
        for phrase, cue in CUE_KEYWORDS.items():
            if is_similar(phrase, text):
                print(f"🎯 유사도 매칭 성공: \"{phrase}\" ≈ \"{text}\"")
                trigger_ma2_cue(cue)

# 실시간 마이크 스트리밍
def main():
    print("🎧 실시간 음성 인식 시스템 시작 (Whisper + VAD + 유사도)...")

    def callback(indata, frames, time, status):
        audio_data = indata[:, 0]
        if is_voice_present(audio_data):
            process_audio(audio_data)
        else:
            print("🤫 (무음 또는 잡음)")

    with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate, blocksize=blocksize):
        input("엔터를 누르면 종료됩니다.\n")

if __name__ == "__main__":
    main()

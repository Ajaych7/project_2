!pip install TTS

from TTS.api import TTS

# Initialize the model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)

# Clone voice (requires reference audio)
tts.tts_to_file(text="Text you want to speak", 
                file_path="output.wav",
                speaker_wav="reference_voice.wav",
                language="en")

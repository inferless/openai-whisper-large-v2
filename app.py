import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import librosa
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class InferlessPythonModel:
    @staticmethod
    def download_audio(audio_url):
        response = requests.get(audio_url, stream=True)
        filename = audio_url.split("/")[-1]

        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception("Error Downloading Audio File")

        return filename
        
    def initialize(self):
        model_id = "openai/whisper-large-v2"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id,device_map="cuda")
        self.model.config.forced_decoder_ids = None

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        audio_file_path = InferlessPythonModel.download_audio(audio_url)
        audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)

        input_features = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

        predicted_ids = self.model.generate(input_features.to("cuda"))
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return {"transcribed_output": transcription[0]}

    def finalize(self):
        self.processor = None
        self.model = None


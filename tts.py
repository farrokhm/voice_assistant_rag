# TextToSpeechService
import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel
#nltk.download()
#nltk.download('punkt')
#nltk.download('punkt_tab')

# Suppress specific warning about deprecated weight normalization
warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class.
        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_9"):
        """
        Synthesizes audio from the given text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")

        # Explicitly create attention mask if necessary
        if "attention_mask" not in inputs:
            # You may need to handle pad_token_id differently
            pad_token_id = (
                self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id
                else 0
            )
            inputs["attention_mask"] = (inputs["input_ids"] != pad_token_id).long()

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Provide a default or manually set pad_token_id if necessary
            pad_token_id = (
                self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id
                else 0
            )
            audio_array = self.model.generate(**inputs, pad_token_id=pad_token_id)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_9"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_9".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)

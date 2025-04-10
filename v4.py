def synthesize_voice(model, input_spectrogram):
    """Convert spectrogram back to audio"""
    model.eval()
    with torch.no_grad():
        # Convert input to tensor
        input_tensor = torch.tensor(input_spectrogram, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Generate output spectrogram
        output = model(input_tensor).cpu().numpy().squeeze()
        
        # Convert mel-spectrogram back to audio
        audio = librosa.feature.inverse.mel_to_audio(
            output, 
            sr=22050,  # sample rate
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        return audio

# Example usage
sample_input = X_val[0]  # take first validation sample
generated_audio = synthesize_voice(model, sample_input)

# Save generated audio
sf.write("generated_voice.wav", generated_audio, 22050)

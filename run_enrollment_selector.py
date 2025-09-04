import gradio as gr

def select_audio(audio, start, end):
    if audio is None:
        return None, None, None
    sr, audio_data = audio
    if end > start:
        if start - int(start) >= 0.6:
            raise ValueError("Start time's decimal part must be less than 0.6 to represent seconds correctly.")
        if end - int(end) >= 0.6:
            raise ValueError("End time's decimal part must be less than 0.6 to represent seconds correctly.")
        start_secs = 60 * int(start) + 100 * (start - int(start))
        start_samples = int(start_secs * sr)
        end_secs = 60 * int(end) + 100 * (end - int(end))
        end_samples = int(end_secs * sr)
        audio_data = audio_data[start_samples:end_samples]
    if audio_data.ndim == 1:
        return (sr, audio_data), None, None
    else:
        return (sr, audio_data), (sr, audio_data[:, 0]), (sr, audio_data[:, 1])

if __name__ == "__main__":
    interface = gr.Interface(
        fn=select_audio,
        inputs=[
            gr.Audio(label="Input Audio"),
            gr.Number(0, minimum=0, label="Start"),
            gr.Number(0, minimum=0, label="End"),

        ], 
        outputs=[
            gr.Audio(label="Selected Audio"),
            gr.Audio(label="Selected Audio (channel 1)"),
            gr.Audio(label="Selected Audio (channel 2)"),
        ],
        allow_flagging='never',
    )
    interface.launch()
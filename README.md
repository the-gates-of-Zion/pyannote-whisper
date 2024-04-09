# pyannote-whisper

Run ASR and speaker diarization based on whisper and pyannote.audio.

## Installation
1. Install whisper.
2. Install pyannote.audio.
3. Install speechbrain


## Command-line usage

Same as whisper except a new param `diarization`:

    python -m pyannote_whisper.cli.transcribe data/afjiv.wav --model tiny --diarization True

please find more details in [this](https://gist.github.com/hbredin/049f2b629700bcea71324d2c1e7f8337) notebook.

```python
import whisper
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote_whisper.utils import diarize_text
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="your/token")
model = whisper.load_model("tiny.en")
diarization_result = pipeline("data/afjiv.wav")

from pyannote.audio import Audio
audio = Audio(sample_rate=16000, mono=True)
audio_file = "data/afjiv.wav"
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    waveform, sample_rate = audio.crop(audio_file, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")

```
## Meeting summary from chatgpt

```
summarize the following words: SPEAKER_00  I think if you're a leader and you don't understand the terms that you're using, that's probably the first start.
SPEAKER_00  It's really important that as a leader in the organisation you understand what digitisation means.
SPEAKER_00  You take the time to read widely in the sector.
SPEAKER_00  There are a lot of really good books, Kevin Kelly, who started Wired magazine has written a great book on various technologies.
...
```
```
Leaders need to understand the terms and technologies of digitization in order to separate hype from hope and apply it effectively to their business. Becoming data-driven and collaborating with technology vendors can help organizations achieve their desired outcomes. Procurement processes need to be fundamentally rethought to improve functionality and eliminate unnecessary spending.
```

```
summarize the opinions of different speaker
```

```
Speaker 00 emphasizes the importance of leaders understanding digitization terms and technologies to apply them effectively to their business. 
Speaker 01 suggests that taking control of spend data is a crucial first step in digital transformation. 
Speaker 02 believes that procurement processes need to be rethought to improve functionality and eliminate unnecessary spending. 
Speaker 03 advises understanding the purpose of digitization and collaborating with colleagues for a better outcome. 
Speaker 04 suggests thinking big and being brave, as well as talking to technology vendors to achieve desired outcomes.
```
## make it offline
https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb

## start
python pyannote_whisper/transcribe.py data/afjiv.wav --model tiny --diarization True
### start
python pyannote_whisper/transcribe.py data/afjiv.wav --model large-v3 --diarization True

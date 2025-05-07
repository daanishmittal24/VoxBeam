# 🚀 VoxBeam: Control Your PC with Your Voice! 🎙️💻

<!-- SHOWOFF: VoxBeam Demo Animation -->
![VoxBeam Demo Animation](assets/images/voxbeam_demo.gif)

```ascii
   🎤                    🧠                     ⚡
[Input] -----> [Whisper Model] -----> [Command Action]
   |              |    |               |
   |              |    |               |
   v              v    v               v
[Audio] --> [Processing] --> [Match] --> [Execute]
```

Welcome to **VoxBeam** – the next-generation, AI-powered voice command platform for your computer. VoxBeam transforms your spoken words into powerful actions, letting you control your PC, launch apps, and automate workflows hands-free.

> **Experience the future of productivity.**

---

## ✨ VoxBeam Highlights

- 🧠 **Powered by OpenAI Whisper**: Industry-leading speech recognition for unmatched accuracy.
- 🎤 **Effortless Command Recording**: Record single or multiple samples for robust, personalized training.
- 🦾 **Synthetic Data Augmentation**: VoxBeam automatically creates pitch, speed, and noise variations for ultra-resilient models.
- ⚡ **Real-Time Execution**: Instant command recognition and action.
- 🖥️ **Intuitive GUI**: Simple, modern interface for recording, training, and testing.
- 🔊 **Advanced Audio Pipeline**: Noise reduction, silence trimming, normalization, and more.
- 🚀 **GPU-Accelerated**: Whisper runs lightning-fast on CUDA GPUs (CPU fallback supported).
- 🛠️ **Extensible**: Add your own commands and actions with ease.

---

## 🎬 SHOWOFF: VoxBeam in Action

```ascii
Recording Mode Demo
┌────────────────────────────────────┐
│        🎙️ Record Command           │
│ ┌────────────────────────────────┐ │
│ │        [Start Recording]       │ │
│ │     ▁▂▃▅▂▁▂▃▄▅▄▃▂▁▂▃▂▁        │ │
│ │        [Stop Recording]        │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘
```

- **VoxBeam in Action:**
  ![VoxBeam Full Demo](assets/images/full_demo.gif)
- **Recording Mode:**
  ![Recording Mode Animation](assets/images/recording_mode.gif)
- **Training Mode:**
  ![Training Mode Animation](assets/images/training_mode.gif)
- **Testing Mode:**
  ![Testing Mode Animation](assets/images/testing_mode.gif)

---

## 🤩 What Makes VoxBeam Unique?

- **Personalized AI**: Trains on your voice and your commands.
- **Synthetic Data Magic**: Augments your data for real-world robustness.
- **Multi-Stage Matching**: Exact, keyword, and fuzzy logic for reliable recognition.
- **Offline-First**: All processing is local after setup.
- **Open Source & Hackable**: Build your own automations and integrations.

---

## 🧠 Model & Theory

```ascii
Whisper Model Pipeline
   [Audio Input]
        ↓
   [Preprocessing]
        ↓
┌─────────────────┐
│  Whisper Model  │
│  ┌───────────┐  │
│  │ Encoder   │  │
│  └───────────┘  │
│       ↓         │
│  ┌───────────┐  │
│  │ Decoder   │  │
│  └───────────┘  │
└─────────────────┘
        ↓
   [Text Output]
```

### Model: OpenAI Whisper
- **Architecture**: Transformer-based encoder-decoder, trained on 680k hours of multilingual and multitask supervised data.
- **Default Model**: `base` (can be swapped for `tiny`, `small`, `medium`, `large` for different speed/accuracy tradeoffs).
- **Input**: 16kHz mono WAV audio.
- **Output**: Transcribed text, optimized for short command phrases.
- **Prompt Engineering**: VoxBeam uses a dynamic prompt listing all available commands to bias the model toward your command set.

### Theory: How VoxBeam Recognizes Commands
1. **Audio Preprocessing**: Cleans and normalizes your voice input for best model performance.
2. **Data Augmentation**: During training, each sample is synthetically varied (pitch, speed, noise, etc.) to simulate real-world conditions and improve generalization.
3. **Speech-to-Text**: Whisper transcribes the audio using a custom prompt.
4. **Command Matching**: VoxBeam uses a 3-stage matching system:
   - **Exact**: Direct string match.
   - **Keyword**: Overlap of important words.
   - **Fuzzy**: Partial/approximate match scoring.
5. **Action Execution**: If a match is found, the mapped action is triggered.

---

## ⚙️ Configuration & Customization

- **Model Selection**: Change the Whisper model size in `model_trainer.py` for different hardware or accuracy needs.
- **Augmentation Settings**: Adjust the number and type of augmentations in `audio_augmenter.py`.
- **Command List**: Add/remove commands by recording new samples and retraining.
- **Action Mapping**: Map new commands to actions in `command_executor.py`.
- **Device Selection**: VoxBeam auto-detects CUDA GPU; falls back to CPU if unavailable.

---

## 🔁 Retraining & Transfer Learning

```ascii
Data Augmentation Pipeline
Original Audio → [Pitch Shift] → [Speed Mod] → [Add Noise]
     ↓              ↓              ↓             ↓
     └──────────────────────┬─────────────────────
                           ↓
                    [Training Data]
                           ↓
                    [Whisper Model]
```

- **Retrain Anytime**: Add new commands or more samples, then retrain via the GUI.
- **Transfer Learning**: While VoxBeam uses Whisper as a frozen base, you can extend it for full transfer learning (fine-tuning Whisper on your own data) with advanced scripts and more compute.
- **Augmentation for Retraining**: Each retrain run generates new synthetic data, making the model more robust over time.
- **Model Checkpoints**: (Planned) Save and load custom fine-tuned models for different users or environments.

---

## 🖥️ GPU & Performance

- **CUDA Acceleration**: If an NVIDIA GPU is detected, all Whisper inference and training run on GPU for massive speedups.
- **CPU Fallback**: Runs on CPU if no GPU is available (slower, but works everywhere).
- **Batch Processing**: Training and augmentation are optimized for parallelism.
- **Progress Feedback**: GUI shows real-time progress bars and status updates during heavy operations.

---

## 🛠️ System Architecture

```ascii
                     VoxBeam Architecture
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Recording   │     │   Training   │     │ Deployment   │
│    Mode      │────>│    Mode      │────>│    Mode      │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       v                    v                    v
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Audio      │     │ Augmentation │     │  Real-time   │
│  Samples     │────>│    Engine    │     │ Recognition  │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                    │
                           v                    v
                    ┌──────────────┐     ┌──────────────┐
                    │   Whisper    │     │   Command    │
                    │   Model      │────>│  Execution   │
                    └──────────────┘     └──────────────┘
```

1. 🎙️ **Voice Input** → 2. 🧹 **Preprocessing** → 3. 🧬 **Augmentation** → 4. 🤖 **Whisper STT** → 5. 🧩 **Command Matching** → 6. ⚙️ **Action Execution**

![System Architecture](assets/images/system_architecture.png)

---

## 📋 Requirements

- **Python** 3.8+
- **Windows** (other OS : core features portable)
- **Microphone**
- **NVIDIA GPU (CUDA)** (optional, for speed)

---

## ⚡ Quickstart

1. **Clone VoxBeam:**
   ```bash
   git clone https://github.com/daanishmittal24/voxbeam.git
   cd voxbeam
   ```
2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # or source venv/bin/activate
   ```
3. **Install Requirements:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Launch VoxBeam:**
   ```bash
   python startup.py
   ```

---

## 🕹️ Modes & Usage

### 🎙️ Recording Mode
- Record commands (single/bulk)
- Organize and review samples
- **SHOWOFF:** <!-- ![Recording Mode](link_to_recording_mode.gif) -->

### 🧠 Training Mode
- Train Whisper on your commands
- Synthetic augmentation for each sample
- **SHOWOFF:** <!-- ![Training Mode](link_to_training_mode.gif) -->

### 🧪 Testing Mode
- Test and execute commands live
- See recognition and action feedback
- **SHOWOFF:** <!-- ![Testing Mode](link_to_testing_mode.gif) -->

### 🏃 Minimal Run
- Lightweight interface for quick command execution
- Accuracy testing tab

---

## 🧬 Technical Deep Dive

- **Audio:** 16kHz mono WAV, DC removal, normalization, high-pass, silence trim, padding
- **Augmentation:** Pitch, speed, volume, noise, shift, reverb
- **Recognition:** Whisper (base), prompt engineering, multi-level matching
- **GUI:** Tkinter, threaded for responsiveness
- **Config:** All major settings are Python variables for easy hacking

---

## 📂 Project Structure

```
VoxBeam/
├── recordings/
├── command_data/
├── trained_models/
├── models/
├── audio_augmenter.py
├── ...
```

---

## 💡 Troubleshooting

```ascii
Common Issues Flowchart
[Start] → No Audio? → Check Microphone
   ↓          ↓           ↓
No Model? → Train → Check GPU → CUDA OK?
   ↓          ↓           ↓         ↓
Low ACC? → More Samples → Retrain → [End]
```

- **No Trained Model?** Run Training Mode after recording.
- **Low Accuracy?** Record more samples, speak clearly, use augmentation.
- **CUDA Issues?** Check drivers and PyTorch install.

---

## 🌱 Roadmap
- Full Whisper fine-tuning
- Contextual & multi-user profiles
- Web UI
- Cross-platform actions

---

## 🤝 Contributing
- Fork, branch, PR – all welcome!

---

## 🙏 Credits
- OpenAI Whisper, PyTorch, Librosa, and all contributors.

---

**VoxBeam: Your voice, your command.**

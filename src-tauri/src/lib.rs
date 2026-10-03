use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::State;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Shared state between Tauri IPC handlers and thread callbacks
pub struct AppState {
    pub is_recording: Arc<Mutex<bool>>,
    pub audio_buffer: Arc<Mutex<Vec<f32>>>,
    pub stream: Arc<Mutex<Option<cpal::Stream>>>,
    pub sample_rate: Arc<Mutex<u32>>,
    pub channels: Arc<Mutex<u16>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            is_recording: Arc::new(Mutex::new(false)),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            stream: Arc::new(Mutex::new(None)),
            sample_rate: Arc::new(Mutex::new(44100)),
            channels: Arc::new(Mutex::new(1)),
        }
    }
}

/// Helper function to resample multi-channel/arbitrary rate audio down to 16kHz mono f32
fn resample_to_16k_mono(input: &[f32], src_rate: u32, channels: u16) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Convert to Mono if input is stereo/multi-channel
    let mono_samples: Vec<f32> = if channels > 1 {
        let channel = channels as usize;
        input
            .chunks(channel)
            .map(|chunk| chunk.iter().sum::<f32>() / (channel as f32))
            .collect()
    } else {
        input.to_vec()
    };

    if src_rate == 16000 {
        return mono_samples;
    }

    // Linear Interpolation Resampling to 16,000 Hz
    let target_rate = 16000.0;
    let ratio = src_rate as f32 / target_rate;
    let output_len = (mono_samples.len() as f32 / ratio).floor() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_index = i as f32 * ratio;
        let index_floor = src_index.floor() as usize;
        let index_ceil = (index_floor + 1).min(mono_samples.len() - 1);
        let weight = src_index - (index_floor as f32);

        let interpolated =
            (1.0 - weight) * mono_samples[index_floor] + weight * mono_samples[index_ceil];
        output.push(interpolated);
    }

    output
}

#[tauri::command]
async fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    let mut recording_flag = state
        .is_recording
        .lock()
        .map_err(|e| format!("Failed to lock state: {}", e))?;

    if *recording_flag {
        return Err("Already recording".to_string());
    }

    // Clear previous audio buffer
    let mut buffer = state
        .audio_buffer
        .lock()
        .map_err(|e| format!("Failed to lock buffer: {}", e))?;
    buffer.clear();

    // Query microphone hardware via CPAL
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "No default audio input device found".to_string())?;

    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to fetch default input config: {}", e))?;

    let sample_rate = config.sample_rate();
    let channels = config.channels();

    *state
        .sample_rate
        .lock()
        .map_err(|e| format!("Failed to set sample rate: {}", e))? = sample_rate;
    *state
        .channels
        .lock()
        .map_err(|e| format!("Failed to set channels: {}", e))? = channels;

    let audio_buffer_clone = Arc::clone(&state.audio_buffer);

    // Build low-level audio input stream
    let err_fn = |err| eprintln!("An error occurred on the audio input stream: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config.into(),
            move |data: &[f32], _: &_| {
                if let Ok(mut buf) = audio_buffer_clone.lock() {
                    buf.extend_from_slice(data);
                }
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config.into(),
            move |data: &[i16], _: &_| {
                if let Ok(mut buf) = audio_buffer_clone.lock() {
                    let float_samples = data.iter().map(|&s| s as f32 / 32768.0);
                    buf.extend(float_samples);
                }
            },
            err_fn,
            None,
        ),
        format => return Err(format!("Unsupported sample format: {:?}", format)),
    }
    .map_err(|e| format!("Failed to build input stream: {}", e))?;

    stream
        .play()
        .map_err(|e| format!("Failed to start stream: {}", e))?;

    *state
        .stream
        .lock()
        .map_err(|e| format!("Failed to store stream: {}", e))? = Some(stream);
    *recording_flag = true;

    println!(
        "Microphone input stream started (Native Rate: {}Hz, Channels: {})",
        sample_rate, channels
    );
    Ok(())
}

#[tauri::command]
async fn stop_recording_and_transcribe(state: State<'_, AppState>) -> Result<String, String> {
    // Stop Recording Stream
    {
        let mut recording_flag = state
            .is_recording
            .lock()
            .map_err(|e| format!("Failed to lock state: {}", e))?;

        if !*recording_flag {
            return Err("Not currently recording".to_string());
        }

        let mut stream_guard = state
            .stream
            .lock()
            .map_err(|e| format!("Failed to lock stream: {}", e))?;

        // Dropping or taking Option::None closes the cpal stream gracefully
        *stream_guard = None;
        *recording_flag = false;
    }

    // Fetch recorded samples
    let raw_samples = {
        let mut buf = state
            .audio_buffer
            .lock()
            .map_err(|e| format!("Failed to lock buffer: {}", e))?;
        std::mem::take(&mut *buf)
    };

    let src_rate = *state
        .sample_rate
        .lock()
        .map_err(|e| format!("Failed to read sample rate: {}", e))?;
    let channels = *state
        .channels
        .lock()
        .map_err(|e| format!("Failed to read channels: {}", e))?;

    if raw_samples.is_empty() {
        return Err("No audio captured from microphone".to_string());
    }

    println!(
        "Processing {} raw audio samples...",
        raw_samples.len()
    );

    // Resample audio down to 16kHz mono
    let pcm_16k = resample_to_16k_mono(&raw_samples, src_rate, channels);

    println!(
        "Audio resampled to 16kHz mono (Total samples: {})",
        pcm_16k.len()
    );

    // Calculate the Root Mean Square (volume level) of the audio
    let mut sum_squares = 0.0;
    for &sample in &pcm_16k {
        sum_squares += sample * sample;
    }
    let root_mean_square = (sum_squares / pcm_16k.len() as f32).sqrt();

    println!("Audio volume level (RMS): {}", root_mean_square);

    // If Root Mean Square is extremely low, it means the mic is muted, blocked, or the room is dead silent.
    // We skip the ML engine entirely to prevent hallucinations.
    if root_mean_square < 0.001 {
        return Ok("[Silence detected. Please ensure your microphone is working and allowed.]".to_string());
    }

    // Load Whisper Engine
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-naija-q5.bin");

    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found at {}", model_path));
    }

    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(model_path, ctx_params)
        .map_err(|e| format!("Failed to load model context: {}", e))?;

    let mut whisper_state = ctx
        .create_state()
        .map_err(|e| format!("Failed to create state: {}", e))?;

    // It considers 5 different transcriptions simultaneously and picks the most accurate one.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch { 
        beam_size: 5, 
        patience: -1.0 
    });
    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Initial prompt for proper capitalisation and punctuation
    params.set_initial_prompt("Hello, my name is Mahdi. I live in Abuja, Nigeria. Let's record this text.");

    println!("Running Whisper inference on microphone audio...");

    whisper_state
        .full(params, &pcm_16k[..])
        .map_err(|e| format!("Inference execution failed: {}", e))?;

    let num_segments = whisper_state.full_n_segments();
    let mut full_transcription = String::new();

    for i in 0..num_segments {
        if let Some(segment) = whisper_state.get_segment(i) {
            if let Ok(segment_text) = segment.to_str() {
                full_transcription.push_str(&segment_text);
                full_transcription.push(' ');
            }
        }
    }

    let result = full_transcription.trim().to_string();
    if result.is_empty() {
        Ok("[No clear speech detected]".to_string())
    } else {
        Ok(result)
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording_and_transcribe
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
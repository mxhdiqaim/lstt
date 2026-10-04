use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::SampleFormat::Float as HoundFloat;
use hound::WavWriter;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::ShellExt;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

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

fn resample_to_16k_mono(input: &[f32], src_rate: u32, channels: u16) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mono_samples: Vec<f32> = if channels > 1 {
        let channel = channels as usize;
        input.chunks(channel).map(|chunk| chunk.iter().sum::<f32>() / (channel as f32)).collect()
    } else {
        input.to_vec()
    };

    if src_rate == 16000 {
        return mono_samples;
    }

    let target_rate = 16000.0;
    let ratio = src_rate as f32 / target_rate;
    let output_len = (mono_samples.len() as f32 / ratio).floor() as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_index = i as f32 * ratio;
        let index_floor = src_index.floor() as usize;
        let index_ceil = (index_floor + 1).min(mono_samples.len() - 1);
        let weight = src_index - (index_floor as f32);
        output.push((1.0 - weight) * mono_samples[index_floor] + weight * mono_samples[index_ceil]);
    }
    output
}

fn format_with_qwen(raw_text: &str, models_dir: &std::path::Path) -> Result<String, String> {
    let backend = LlamaBackend::init().map_err(|e| format!("LLM backend error: {}", e))?;
    let qwen_path = models_dir.join("qwen2.5-0.5b-instruct-q5_k_m.gguf");

    let model = LlamaModel::load_from_file(&backend, qwen_path.to_str().unwrap(), &LlamaModelParams::default())
        .map_err(|e| format!("Failed to load Qwen: {}", e))?;

    let mut ctx_params = LlamaContextParams::default();
    ctx_params = ctx_params.with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    let mut ctx = model.new_context(&backend, ctx_params)
        .map_err(|e| format!("Context error: {}", e))?;

    let prompt = format!(
        "<|im_start|>system\nYou are an expert transcription editor. Format the following raw dictated text with proper punctuation, capitalization, and markdown lists. CRITICAL: Strictly use formal British English spelling conventions (e.g. 'quantisation', 'colour') as is standard in Nigeria. Do not change the original meaning, slang, or language of the text. Output ONLY the formatted text.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        raw_text
    );

    let tokens_list = model.vocab().tokenize(prompt.as_bytes(), false, true);
    let mut batch = LlamaBatch::new(2048, 1);
    let last_index = tokens_list.len() - 1;
    for (i, &token) in tokens_list.iter().enumerate() {
        batch.add(token, i as i32, &[0], i == last_index).unwrap();
    }

    ctx.decode(&mut batch).map_err(|e| format!("Decode failed: {}", e))?;

    let mut formatted_output = String::new();
    let mut n_cur = batch.n_tokens();
    let mut sampler = LlamaSampler::greedy();

    for _ in 0..1024 {
        let token_id = sampler.sample(&ctx, batch.n_tokens() - 1);
        if model.vocab().is_eog(token_id) {
            break;
        }

        let token_bytes = model.vocab().token_to_piece(token_id, false, None);
        let token_str = String::from_utf8_lossy(&token_bytes);

        if token_str == "<|im_end|>" {
            break;
        }

        formatted_output.push_str(&token_str);
        sampler.accept(token_id);

        batch.clear();
        batch.add(token_id, n_cur, &[0], true).unwrap();
        ctx.decode(&mut batch).unwrap();
        n_cur += 1;
    }

    Ok(formatted_output.trim().to_string())
}

#[tauri::command]
async fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    let mut recording_flag = state.is_recording.lock().unwrap();
    if *recording_flag { return Err("Already recording".to_string()); }

    state.audio_buffer.lock().unwrap().clear();

    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No input device")?;
    let config = device.default_input_config().unwrap();

    *state.sample_rate.lock().unwrap() = config.sample_rate();
    *state.channels.lock().unwrap() = config.channels();

    let audio_buffer_clone = Arc::clone(&state.audio_buffer);
    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config.into(),
            move |data: &[f32], _: &_| { audio_buffer_clone.lock().unwrap().extend_from_slice(data); },
            err_fn, None
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config.into(),
            move |data: &[i16], _: &_| {
                audio_buffer_clone.lock().unwrap().extend(data.iter().map(|&s| s as f32 / 32768.0));
            },
            err_fn, None
        ),
        _ => return Err("Unsupported format".to_string()),
    }.unwrap();

    stream.play().unwrap();
    *state.stream.lock().unwrap() = Some(stream);
    *recording_flag = true;
    Ok(())
}

#[tauri::command]
async fn stop_recording_and_transcribe(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    {
        let mut recording_flag = state.is_recording.lock().unwrap();
        if !*recording_flag {
            return Err("Not recording".to_string());
        }
        *state.stream.lock().unwrap() = None;
        *recording_flag = false;
    }

    let raw_samples = std::mem::take(&mut *state.audio_buffer.lock().unwrap());
    let src_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();
    let pcm_16k = resample_to_16k_mono(&raw_samples, src_rate, channels);

    let temp_wav_path = std::env::temp_dir().join("lstt_temp_audio.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: HoundFloat,
    };

    let mut writer = WavWriter::create(&temp_wav_path, spec).unwrap();
    for &sample in &pcm_16k { writer.write_sample(sample).unwrap(); }
    writer.finalize().unwrap();

    let models_dir = app.path().resource_dir().unwrap().join("models");

    // Run Whisper in the Sidecar
    let sidecar_command = app.shell().sidecar("sidecar").map_err(|e| e.to_string())?
        .arg(temp_wav_path.to_str().unwrap())
        .arg(models_dir.to_str().unwrap());

    let output = sidecar_command.output().await.map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(temp_wav_path);

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    let raw_text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw_text.is_empty() || raw_text == "[No clear speech detected]" {
        return Ok(raw_text);
    }

    // Format with LLaMA directly in the main Tauri App
    // We wrap this in spawn_blocking so it doesn't freeze the Tauri async runtime thread
    let models_dir_clone = models_dir.clone();
    let formatted_text = tauri::async_runtime::spawn_blocking(move || {
        format_with_qwen(&raw_text, &models_dir_clone)
    }).await.map_err(|e| e.to_string())??;

    Ok(formatted_text)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![start_recording, stop_recording_and_transcribe])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
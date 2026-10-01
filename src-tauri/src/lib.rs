use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::path::Path;

#[tauri::command]
async fn transcribe_test_file() -> Result<String, String> {
    println!("Frontend requested transcription. Booting engine...");

    // We moved the models folder inside src-tauri
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-naija-q5.bin");

    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found at {}", model_path));
    }

    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(model_path, ctx_params)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let mut state = ctx.create_state()
        .map_err(|e| format!("Failed to create state: {}", e))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Read test.wav from the src-tauri root
    let test_wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test.wav");
    let mut reader = hound::WavReader::open(test_wav_path)
        .map_err(|e| format!("Failed to open test.wav: {}", e))?;

    let spec = reader.spec();
    let audio_data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            reader.samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / 32768.0)
                .collect()
        },
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .map(|s| s.unwrap_or(0.0))
                .collect()
        },
    };

    println!("Audio loaded. Running inference...");

    state.full(params, &audio_data[..])
        .map_err(|e| format!("Transcription failed: {}", e))?;

    let num_segments = state.full_n_segments();
    let mut full_transcription = String::new();

    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            if let Ok(segment_text) = segment.to_str() {
                full_transcription.push_str(&segment_text);
                full_transcription.push(' '); 
            }
        }
    }

    println!("Transcription complete. Sending to frontend...");
    Ok(full_transcription.trim().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![transcribe_test_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
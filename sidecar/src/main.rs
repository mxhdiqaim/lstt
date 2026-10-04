use hound::WavReader;
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: sidecar <path_to_wav> <models_dir>");
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let models_dir = PathBuf::from(&args[2]);

    // Read the audio
    let mut reader = WavReader::open(wav_path).unwrap();
    let samples: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();

    // Run Whisper
    let whisper_path = models_dir.join("ggml-naija-q5.bin");
    let ctx = WhisperContext::new_with_params(
        whisper_path.to_str().unwrap(),
        WhisperContextParameters::default()
    ).unwrap();

    let mut whisper_state = ctx.create_state().unwrap();
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0
    });
    params.set_language(Some("en"));

    // Disable stdout printing from Whisper internals
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    whisper_state.full(params, &samples[..]).unwrap();

    let mut raw_text = String::new();
    for i in 0..whisper_state.full_n_segments() {
        if let Some(segment) = whisper_state.get_segment(i) {
            raw_text.push_str(&segment.to_str().unwrap());
            raw_text.push(' ');
        }
    }

    // ONLY print the final transcription to stdout so Tauri can capture it
    print!("{}", raw_text.trim());
}
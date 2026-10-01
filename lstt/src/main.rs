// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::path::Path;

fn main() {
    println!("--- LSTT Engine Starting ---");

    // Define the path to our quantized model
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-naija-q5.bin");

    if !Path::new(model_path).exists() {
        eprintln!("CRITICAL ERROR: Model file not found at {}", model_path);
        return;
    }

    println!("Found model! Loading into memory...");

    // Initialize the model context
    let ctx_params = WhisperContextParameters::default();
    
    let ctx = match WhisperContext::new_with_params(model_path, ctx_params) {
        Ok(context) => {
            println!("SUCCESS: Nigerian Accented English model loaded efficiently!");
            context
        },
        Err(error) => {
            eprintln!("Failed to load model: {}", error);
            return;
        }
    };

    // Create a state for execution
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create whisper state: {}", e);
            return;
        }
    };

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(true);

    println!("Engine is fully primed. Loading test.wav...");

    // Safely open the audio file
    let mut reader = match hound::WavReader::open("test.wav") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open 'test.wav': {}", e);
            return;
        }
    };

    let spec = reader.spec();

    // Safely extract and format the audio samples
    let audio_data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            // Collect the samples into a Result to catch any reading errors mid-stream
            let samples_result: Result<Vec<f32>, _> = reader
                .samples::<i16>()
                .map(|s| s.map(|val| val as f32 / 32768.0)) // Normalize while handling inner Result
                .collect();
                
            match samples_result {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error parsing integer audio data: {}", e);
                    return;
                }
            }
        },
        hound::SampleFormat::Float => {
            let samples_result: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            match samples_result {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error parsing float audio data: {}", e);
                    return;
                }
            }
        },
    };

    println!("Audio loaded safely. Starting transcription...");

    // Safely run inference
    match state.full(params, &audio_data[..]) {
        Ok(_) => println!("Transcription execution finished."),
        Err(e) => {
            eprintln!("Transcription engine failed: {}", e);
            return;
        }
    }

    // Safely get the number of text segments
    let num_segments = state.full_n_segments();
    
    println!("\n--- TRANSCRIPTION RESULT ---");
    for i in 0..num_segments {
        // Safely extract each individual segment
        match state.get_segment(i) {
            Some(segment) => match segment.to_str() {
                Ok(segment_text) => println!("{}", segment_text),
                Err(e) => eprintln!("Failed to read segment {}: {}", i, e),
            }
            None => eprintln!("Failed to read segment {}: out of bounds", i),
        }
    }
    println!("----------------------------\n");
}
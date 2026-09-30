// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::path::Path;

fn main() {
    println!("--- LSTT Engine Starting ---");

    // Define the path to our quantized model
    let model_path = "../models/ggml-naija-q5.bin";

    if !Path::new(model_path).exists() {
        eprintln!("CRITICAL ERROR: Model file not found at {}", model_path);
        return;
    }

    println!("Found model! Loading into memory...");

    // Initialize the model context
    let ctx_params = WhisperContextParameters::default();
    
    // We bind the context to _ctx. 
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
    // The state manages the scratch buffers and context windows required during transcription.
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            eprint!("Failed to create whisper state: {}", e);
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

    println!("Engine is fully primed and ready for PCM audio streams!");
}
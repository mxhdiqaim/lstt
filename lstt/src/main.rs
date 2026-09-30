// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use whisper_rs::{WhisperContext, WhisperContextParameters};
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
    let _ctx = match WhisperContext::new_with_params(model_path, ctx_params) {
        Ok(context) => {
            println!("SUCCESS: Nigerian Accented English model loaded efficiently!");
            context
        },
        Err(error) => {
            eprintln!("Failed to load model: {}", error);
            return;
        }
    };

    println!("Engine test complete. Model is ready for audio!");
}
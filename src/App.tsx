import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import "./App.css";

function App() {
  const [transcription, setTranscription] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  const handleTestTranscription = async () => {
    setIsProcessing(true);
    setTranscription("Loading model and transcribing test.wav...");
    
    try {
      // This directly calls the #[tauri::command] async fn transcribe_test_file() in lib.rs
      const text = await invoke<string>("transcribe_test_file");
      setTranscription(text);
    } catch (error) {
      // Rust Err() returns map to JavaScript rejected promises
      console.error("Rust Backend Error:", error);
      setTranscription(`Error: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 bg-slate-900">
      <div className="max-w-2xl w-full flex flex-col gap-8">
        
        <div className="text-center">
          <h1 className="text-3xl font-bold text-slate-100 mb-2">
            Local Speech-to-Text
          </h1>
        </div>

        <button
          onClick={handleTestTranscription}
          disabled={isProcessing}
          className={`px-6 py-3 rounded-lg font-semibold text-white transition-all duration-200 
            ${isProcessing 
              ? "bg-indigo-500/50 cursor-not-allowed" 
              : "bg-indigo-600 hover:bg-indigo-500 active:scale-95"
            }`}
        >
          {isProcessing ? "Transcribing Engine..." : "Run Test Transcription (JFK)"}
        </button>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 min-h-[150px] shadow-lg">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Output
          </h3>
          <p className="text-lg leading-relaxed text-slate-200 whitespace-pre-wrap">
            {transcription || "Awaiting audio input..."}
          </p>
        </div>
        
      </div>
    </div>
  );
}

export default App;

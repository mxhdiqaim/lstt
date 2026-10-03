import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

function App() {
  const [transcription, setTranscription] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  const handleToggleRecording = async () => {
    if (!isRecording) {
      try {
        await invoke("start_recording");
        setIsRecording(true);
        setTranscription("Listening... Speak into your microphone.");
      } catch (error) {
        console.error("Failed to start recording:", error);
        setTranscription(`Error: ${error}`);
      }
    } else {
      setIsRecording(false);
      setIsProcessing(true);
      setTranscription("Processing speech and running model inference...");

      try {
        const result = await invoke<string>("stop_recording_and_transcribe");
        setTranscription(result);
      } catch (error) {
        console.error("Transcription error:", error);
        setTranscription(`Error: ${error}`);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 bg-slate-900">
      <div className="max-w-2xl w-full flex flex-col gap-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-slate-100 mb-2">
            Local Speech-to-Text Engine
          </h1>
          <p className="text-slate-400">
            Offline speech recognition powered by Rust & GGML
          </p>
        </div>

        <div className="flex justify-center">
          <button
            onClick={handleToggleRecording}
            disabled={isProcessing}
            className={`px-8 py-4 rounded-full font-semibold text-white transition-all duration-200 flex items-center gap-3 shadow-lg ${
              isProcessing
                ? "bg-slate-700 cursor-not-allowed opacity-60"
                : isRecording
                ? "bg-red-600 hover:bg-red-500 animate-pulse"
                : "bg-indigo-600 hover:bg-indigo-500 active:scale-95"
            }`}
          >
            <span
              className={`w-3 h-3 rounded-full ${
                isRecording ? "bg-white" : "bg-red-400"
              }`}
            />
            {isProcessing
              ? "Processing Audio..."
              : isRecording
              ? "Stop Recording & Transcribe"
              : "Start Recording"}
          </button>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 min-h-[180px] shadow-lg">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Live Output
          </h3>
          <p className="text-lg leading-relaxed text-slate-200 whitespace-pre-wrap">
            {transcription || "Click 'Start Recording' and speak into your microphone..."}
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;

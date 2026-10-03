import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

function App() {
  const [transcription, setTranscription] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  // Helper function to copy text to macOS system clipboard
  const copyToClipboard = async (textToCopy: string) => {
    if (!textToCopy || textToCopy.startsWith("[")) return;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy to macOS system clipboard:", err);
    }
  };

  const handleToggleRecording = async () => {
    if (!isRecording) {
      try {
        await invoke("start_recording");
        setIsRecording(true);
        setTranscription("");
      } catch (error) {
        console.error("Failed to start recording:", error);
        setTranscription(`Error: ${error}`);
      }
    } else {
      setIsRecording(false);
      setIsProcessing(true);

      try {
        const result = await invoke<string>("stop_recording_and_transcribe");
        setTranscription(result);
        
        // Auto-copy transcription to macOS system clipboard
        await copyToClipboard(result);
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
      <div className="max-w-2xl w-full flex flex-col gap-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-slate-100 mb-2">
            Local Speech-to-Text Engine
          </h1>
          <p className="text-slate-400">
            Offline speech recognition powered by Rust & GGML
          </p>
        </div>

        {/* Record Button */}
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

        {/* Live Output Section */}
        <div className="bg-slate-800 rounded-xl p-3 border border-slate-700 shadow-lg flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Editable Output
            </h3>
            
            <div className="flex items-center gap-2">
              {copied && (
                <span className="text-xs text-emerald-400 font-medium transition-all">
                  ✓ Copied to macOS Clipboard
                </span>
              )}
              {transcription && (
                <button
                  onClick={() => copyToClipboard(transcription)}
                  className="text-xs px-3 py-1 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded transition-colors"
                >
                  Copy Text
                </button>
              )}
            </div>
          </div>

          {/* Editable Text Area */}
          <textarea
            value={transcription}
            onChange={(e) => setTranscription(e.target.value)}
            disabled={isRecording || isProcessing}
            placeholder={
              isRecording
                ? "Listening... Speak into your microphone."
                : isProcessing
                ? "Processing speech and running model inference..."
                : "Click 'Start Recording' to begin speaking, or type/edit text here..."
            }
            className="w-full h-48 p-4 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y text-base leading-relaxed"
          />
        </div>
      </div>
    </div>
  );
}

export default App;
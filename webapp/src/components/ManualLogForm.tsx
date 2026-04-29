"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ─── Voice Input Hook ────────────────────────────────────────────────────────
function useVoiceInput() {
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(false);
  const recognitionRef = useRef<any>(null);
  const listeningRef = useRef(false); // ref so toggle always sees current value

  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    setSupported(!!SR);
  }, []);

  const toggle = useCallback((onResult: (text: string) => void) => {
    if (listeningRef.current) {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
      listeningRef.current = false;
      setListening(false);
      return;
    }
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (e: any) => {
      onResult(e.results[0][0].transcript);
    };
    rec.onerror = () => { listeningRef.current = false; setListening(false); };
    rec.onend   = () => { listeningRef.current = false; setListening(false); };
    recognitionRef.current = rec;
    rec.start();
    listeningRef.current = true;
    setListening(true);
  }, []);

  return { listening, supported, toggle };
}

// ─── Mic Button ──────────────────────────────────────────────────────────────
function MicButton({ onResult, style }: { onResult: (t: string) => void; style?: React.CSSProperties }) {
  const { listening, supported, toggle } = useVoiceInput();
  if (!supported) return null;
  return (
    <button
      type="button"
      onClick={() => toggle(onResult)}
      title={listening ? "Tap to stop recording" : "Tap to speak"}
      style={{
        background: listening ? "rgba(239,68,68,0.15)" : "rgba(99,102,241,0.1)",
        border: `1px solid ${listening ? "#ef4444" : "rgba(99,102,241,0.3)"}`,
        borderRadius: "50%",
        width: "2rem",
        height: "2rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
        transition: "all 0.2s",
        animation: listening ? "pulse-mic 1s ease-in-out infinite" : "none",
        ...style,
      }}
    >
      {listening ? "⏹" : "🎤"}
    </button>
  );
}

export default function ManualLogForm({ initialDate, userId, onSaved }: { initialDate?: string; userId: number; onSaved?: () => void }) {
  const [selectedDate, setSelectedDate] = useState(initialDate || new Date().toISOString().split('T')[0]);
  const [notes, setNotes] = useState("");
  const [logTime, setLogTime] = useState("");
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (initialDate) setSelectedDate(initialDate);
  }, [initialDate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg("");

    try {
      const logDateToSend = selectedDate;
      const timeToSend = logTime || null;
      const promises: Promise<Response>[] = [];

      // Log notes if provided
      if (notes) {
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-User-ID": userId.toString() },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "note",
            value: 0,
            raw_input: notes
          }),
        }));
      }

      if (promises.length === 0) return;

      setStatus("submitting");
      setStatusMsg("Saving...");
      const results = await Promise.all(promises);
      for (const res of results) {
        if (!res.ok) throw new Error("One or more logs failed to save.");
      }

      setStatus("success");
      setStatusMsg("✓ Logged Successfully");
      setNotes("");
      onSaved?.();
      setTimeout(() => { setStatus("idle"); setStatusMsg(""); }, 3000);
    } catch (err: any) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  };

  const busy = status === "submitting";

  return (
    <div className="premium-card">
      <h2 style={{ fontSize: "1.5rem", marginBottom: "0.5rem", fontWeight: 600 }}>Exercise & Notes Log</h2>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem", fontSize: "0.95rem" }}>
        Log specific workouts or context notes to enrich your Garmin digital twin.
      </p>
      
      {errorMsg && (
        <div style={{ 
          padding: "0.75rem", 
          background: "rgba(239, 68, 68, 0.1)", 
          color: "var(--danger)", 
          borderRadius: "var(--radius-sm)", 
          marginBottom: "1rem",
          fontSize: "0.875rem"
        }}>
          {errorMsg}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "1.5rem", padding: "1rem", background: "rgba(99, 102, 241, 0.05)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(99, 102, 241, 0.1)" }}>
          <label htmlFor="logDate" className="premium-label" style={{ marginBottom: "0.5rem", display: "block" }}>🎯 Log Date (Change for Backfilling)</label>
          <input 
            type="date" 
            id="logDate" 
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="premium-input"
            style={{ width: "100%", padding: "0.75rem", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "#fff" }}
          />
        </div>

        <div style={{ marginTop: "1rem" }}>
          <label htmlFor="log_time" className="premium-label">Specific Time (Optional)</label>
          <input 
            type="time" 
            id="log_time" 
            className="premium-input" 
            value={logTime}
            onChange={(e) => setLogTime(e.target.value)}
          />
        </div>

        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <label htmlFor="notes" className="premium-label" style={{ marginBottom: 0 }}>Exercise & General Notes</label>
            <MicButton onResult={(t) => setNotes(prev => prev ? prev + " " + t : t)} />
          </div>
          <input
            type="text"
            id="notes"
            className="premium-input"
            placeholder="e.g. Ran 5k, feeling energetic"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        <button 
          type="submit" 
          className="premium-button" 
          disabled={busy}
          style={{ 
            marginTop: "1.5rem",
            width: "100%", 
            opacity: busy ? 0.7 : 1,
            cursor: busy ? "not-allowed" : "pointer"
          }}
        >
          {busy ? statusMsg : status === "success" ? statusMsg : "Save Daily Log"}
        </button>
      </form>
    </div>
  );
}

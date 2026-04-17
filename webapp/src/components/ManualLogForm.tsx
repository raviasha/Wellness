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
  const [weight, setWeight] = useState("");
  const [food, setFood] = useState("");
  const [selectedDate, setSelectedDate] = useState(initialDate || new Date().toISOString().split('T')[0]);
 
  useEffect(() => {
    if (initialDate) setSelectedDate(initialDate);
  }, [initialDate]);
  const [notes, setNotes] = useState("");
  const [logTime, setLogTime] = useState("");
  const [parsedNutrition, setParsedNutrition] = useState<any>(null);
  const [status, setStatus] = useState<"idle" | "parsing" | "submitting" | "success" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // Reset parsed result whenever the food text changes
  useEffect(() => {
    setParsedNutrition(null);
  }, [food]);

  const runMagicParse = async (text: string): Promise<any> => {
    const res = await fetch("/api/nutrition/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User-ID": userId.toString() },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("Nutrition parse failed. Check backend logs.");
    return res.json();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg("");

    try {
      const logDateToSend = selectedDate;
      const timeToSend = logTime || null;
      const promises: Promise<Response>[] = [];

      // 1. Auto-run Magic Parse if food is entered
      let nutrition = parsedNutrition;
      if (food) {
        if (!nutrition) {
          setStatus("parsing");
          setStatusMsg("✨ Analyzing nutrition...");
          nutrition = await runMagicParse(food);
          setParsedNutrition(nutrition);
        }
        const calories = nutrition?.calories || 0;
        const payload = JSON.stringify({ text: food, parsed: nutrition || null });
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-User-ID": userId.toString() },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "food",
            value: calories,
            raw_input: payload,
          }),
        }));
      }

      // 2. Log weight if provided
      if (weight) {
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-User-ID": userId.toString() },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "weight",
            value: parseFloat(weight),
            raw_input: `Weight entry: ${weight}`
          }),
        }));
      }

      // 3. Log notes if provided
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
      setWeight("");
      setFood("");
      setNotes("");
      setParsedNutrition(null);
      onSaved?.();
      setTimeout(() => { setStatus("idle"); setStatusMsg(""); }, 3000);
    } catch (err: any) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  };

  const busy = status === "parsing" || status === "submitting";

  return (
    <div className="premium-card">
      <h2 style={{ fontSize: "1.5rem", marginBottom: "0.5rem", fontWeight: 600 }}>Manual Logging</h2>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem", fontSize: "0.95rem" }}>
        Input daily routines like weight, specific workouts, and food ingestion to enrich your digital twin.
        Food entries are automatically parsed for calories and macronutrients.
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

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <div>
            <label htmlFor="weight" className="premium-label">Weight (kg/lbs)</label>
            <input 
              type="number" 
              id="weight" 
              className="premium-input" 
              placeholder="e.g. 75.5" 
              step="0.1"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="log_time" className="premium-label">Specific Time (Optional)</label>
            <input 
              type="time" 
              id="log_time" 
              className="premium-input" 
              value={logTime}
              onChange={(e) => setLogTime(e.target.value)}
            />
          </div>
        </div>

        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <label htmlFor="food" className="premium-label" style={{ marginBottom: 0 }}>Food & Nutrition Today</label>
              <MicButton onResult={(t) => setFood(prev => prev ? prev + " " + t : t)} />
            </div>
            <span style={{ fontSize: "0.7rem", color: "var(--accent-primary)", fontWeight: 600 }}>✨ Auto-parsed on save</span>
          </div>
          <textarea 
            id="food" 
            className="premium-input" 
            placeholder="e.g. 2 eggs, 1 bowl of oats for breakfast, chicken rice for lunch..." 
            rows={3}
            value={food}
            onChange={(e) => setFood(e.target.value)}
            style={{ resize: "vertical" }}
          />

          {parsedNutrition && (
            <div style={{ 
              marginTop: "0.75rem", 
              padding: "0.75rem", 
              background: "rgba(99, 102, 241, 0.05)", 
              border: "1px solid rgba(99, 102, 241, 0.2)",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.85rem"
            }}>
              <p style={{ fontWeight: 600, color: "var(--accent-primary)", marginBottom: "0.25rem" }}>✨ AI Interpretation:</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                <span>Calories: <strong>{parsedNutrition.calories} kcal</strong></span>
                <span>Quality: <strong>{parsedNutrition.quality_score ?? "—"}/10</strong></span>
                <span>Protein: <strong>{parsedNutrition.protein_g}g</strong></span>
                <span>Carbs / Fat: <strong>{parsedNutrition.carbs_g}g / {parsedNutrition.fat_g}g</strong></span>
              </div>
            </div>
          )}
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

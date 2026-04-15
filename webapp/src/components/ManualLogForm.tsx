"use client";

import { useState, useEffect } from "react";

export default function ManualLogForm({ initialDate }: { initialDate?: string }) {
  const [weight, setWeight] = useState("");
  const [food, setFood] = useState("");
  const [selectedDate, setSelectedDate] = useState(initialDate || new Date().toISOString().split('T')[0]);
 
  useEffect(() => {
    if (initialDate) setSelectedDate(initialDate);
  }, [initialDate]);
  const [stress, setStress] = useState(5);
  const [notes, setNotes] = useState("");
  const [logTime, setLogTime] = useState("");
  const [parsedNutrition, setParsedNutrition] = useState<any>(null);
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error" | "parsing">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  const handleMagicParse = async () => {
    if (!food) return;
    setStatus("parsing");
    setErrorMsg("");
    setParsedNutrition(null);
    try {
      const res = await fetch("/api/nutrition/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: food }),
      });
      if (!res.ok) throw new Error("Failed to parse nutrition. Check backend logs.");
      const data = await res.json();
      setParsedNutrition(data);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setStatus("idle");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("submitting");
    setErrorMsg("");
    
    try {
      const promises = [];
      const logDateToSend = selectedDate;
      const timeToSend = logTime || null;

      // 1. Log weight if provided
      if (weight) {
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "weight",
            value: parseFloat(weight),
            raw_input: `Weight entry: ${weight}`
          }),
        }));
      }

      // 2. Log food if provided
      if (food) {
        const calories = parsedNutrition?.calories || 0;
        const payload = parsedNutrition ? JSON.stringify({text: food, parsed: parsedNutrition}) : food;
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "food",
            value: calories,
            raw_input: payload,
          }),
        }));
      }

      // 3. (Removed manual stress logging, relying exclusively on Garmin physiological stress outcomes)

      // 4. Log notes if provided
      if (notes) {
        promises.push(fetch("/api/logs/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            log_date: logDateToSend,
            log_time: timeToSend,
            log_type: "note",
            value: 0,
            raw_input: notes
          }),
        }));
      }

      const results = await Promise.all(promises);
      for (const res of results) {
        if (!res.ok) throw new Error("One or more logs failed to save.");
      }

      setStatus("success");
      setWeight("");
      setFood("");
      setStress(5);
      setNotes("");
      setParsedNutrition(null);
      
      setTimeout(() => setStatus("idle"), 3000);
    } catch (err: any) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  };

  return (
    <div className="premium-card">
      <h2 style={{ fontSize: "1.5rem", marginBottom: "0.5rem", fontWeight: 600 }}>Manual Logging</h2>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem", fontSize: "0.95rem" }}>
        Input daily routines like weight, specific workouts, and food ingestion to enrich your digital twin.
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "0.5rem" }}>
            <label htmlFor="food" className="premium-label" style={{ marginBottom: 0 }}>Food & Nutrition Today</label>
            <button 
              type="button" 
              onClick={handleMagicParse}
              disabled={!food || status === "parsing"}
              style={{ 
                fontSize: "0.75rem", 
                padding: "0.25rem 0.6rem", 
                borderRadius: "var(--radius-sm)",
                background: "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
                color: "white",
                border: "none",
                cursor: "pointer",
                opacity: (!food || status === "parsing") ? 0.5 : 1
              }}
            >
              {status === "parsing" ? "Analyzing..." : "✨ Magic Parse"}
            </button>
          </div>
          <textarea 
            id="food" 
            className="premium-input" 
            placeholder="e.g. 2000 Calories, 150g Protein. 2 apples..." 
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
              <p style={{ fontWeight: 600, color: "var(--accent-primary)", marginBottom: "0.25rem" }}>AI Interpretation:</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                <span>Calories: <strong>{parsedNutrition.calories} kcal</strong></span>
                <span>Type: <strong>{parsedNutrition.nutrition_type}</strong></span>
                <span>Protein: <strong>{parsedNutrition.protein_g}g</strong></span>
                <span>Carbs/Fat: <strong>{parsedNutrition.carbs_g}g / {parsedNutrition.fat_g}g</strong></span>
              </div>
            </div>
          )}
        </div>

        <div style={{ marginTop: "1rem" }}>
          <label htmlFor="notes" className="premium-label">Exercise & General Notes</label>
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
          disabled={status === "submitting" || status === "parsing"}
          style={{ 
            marginTop: "1.5rem",
            width: "100%", 
            opacity: (status === "submitting" || status === "parsing") ? 0.7 : 1,
            cursor: (status === "submitting" || status === "parsing") ? "not-allowed" : "pointer"
          }}
        >
          {status === "submitting" ? "Saving..." : status === "success" ? "✓ Logged Successfully" : "Save Daily Log"}
        </button>
      </form>
    </div>
  );
}

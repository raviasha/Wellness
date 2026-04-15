"use client";

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import UserManual from './UserManual';
import AdminPanel from './AdminPanel';
import ManualLogForm from './ManualLogForm';

const GOALS = [
  { value: "overall_wellness", label: "🎯 Overall Wellness" },
  { value: "longevity_optimization", label: "⏳ Longevity & RHR" },
  { value: "metabolic_health", label: "⚖️ Metabolic & Weight" },
  { value: "recovery_focus", label: "🛌 Recovery & HRV" },
  { value: "muscle_preservation", label: "💪 Muscle Preservation" },
];

const RadialDial = ({ value, min = 0, max = 100, label, color = "var(--accent-primary)", children = null }: any) => {
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const percentage = (Math.max(value, min) - min) / (max - min);
  const strokeDashoffset = circumference - Math.min(percentage, 1) * circumference;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ position: 'relative', width: 120, height: 120 }}>
        <svg width="120" height="120" viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="50" cy="50" r={radius} fill="transparent" stroke="var(--bg-card)" strokeWidth="8" />
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="transparent"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)" }}
          />
        </svg>
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
          <span style={{ fontSize: "1.5rem", fontWeight: "bold", color: "var(--text-primary)" }}>{value}</span>
          {children && <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "-0.2rem" }}>{children}</span>}
        </div>
      </div>
      <span style={{ marginTop: '0.75rem', color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 500 }}>{label}</span>
    </div>
  );
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "1rem", borderRadius: "12px", boxShadow: "0 10px 25px -5px rgba(0,0,0,0.5)" }}>
        <h4 style={{ margin: "0 0 0.5rem 0", color: "var(--accent-primary)", fontSize: "0.9rem" }}>{label}</h4>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.85rem" }}>
          {payload.map((p: any) => (
            <div key={p.dataKey} style={{ color: p.stroke, display: "flex", justifyContent: "space-between", gap: "1rem" }}>
              <span>{p.name}:</span>
              <strong>{typeof p.value === 'number' ? p.value.toFixed(1) : (p.value ?? "—")}</strong>
            </div>
          ))}
        </div>
        {payload.some((p: any) => ['hrv', 'rhr', 'sleep', 'weight'].includes(p.dataKey)) && (
          <div style={{ marginTop: "0.5rem", fontSize: "0.65rem", color: "var(--text-secondary)", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "0.4rem" }}>
            Biomarkers responding to previous day behavior
          </div>
        )}
      </div>
    );
  }
  return null;
};

export default function GarminDashboard() {
  const [users, setUsers] = useState<any[]>([]);
  const [currentUserId, setCurrentUserId] = useState<number>(3);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'manual' | 'settings' | 'admin' | 'evals'>('dashboard');
  
  const [data, setData] = useState<any>(null);
  const [syncStatus, setSyncStatus] = useState<{syncing: boolean, message: string} | null>(null);
  const [simulateMode, setSimulatedMode] = useState(false);
  const [insight, setInsight] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedGoal, setSelectedGoal] = useState("overall_wellness");
  const [inferenceMode, setInferenceMode] = useState("auto");
  const [evals, setEvals] = useState<any[]>([]);
  const [dashboardMetrics, setDashboardMetrics] = useState<any>(null);
  const [backfillDate, setBackfillDate] = useState<string | undefined>(undefined);

  const [creds, setCreds] = useState({ email: "", password: "", name: "" });

  const authHeaders = { 'X-User-ID': currentUserId.toString() };

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchInsight();
    fetchEvals();
    fetchDashboardMetrics();
  }, [currentUserId, selectedGoal, inferenceMode]);
  
  const fetchDashboardMetrics = async () => {
    try {
      const res = await fetch("/api/dashboard/metrics", { headers: authHeaders });
      if (res.ok) setDashboardMetrics(await res.json());
    } catch (e) {}
  };

  const fetchUsers = async () => {
    try {
      const res = await fetch("/api/users");
      const list = await res.json();
      setUsers(list);
      if (list.length > 0) {
        const defaultUser = list.find((u: any) => u.username === 'Tejas');
        setCurrentUserId(defaultUser ? defaultUser.id : list[0].id);
      }
    } catch (e) { console.error(e); }
  };

  const fetchInsight = async () => {
    try {
      const res = await fetch(`/api/recommendations?goal=${selectedGoal}&mode=${inferenceMode}`, { headers: authHeaders });
      if (res.ok) {
        setInsight(await res.json());
      }
    } catch (err) { console.error(err); }
  };

  const fetchEvals = async () => {
    try {
      const res = await fetch("/api/persona/evals", { headers: authHeaders });
      if (res.ok) setEvals(await res.json());
    } catch (err) { console.error(err); }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/history", { headers: authHeaders });
      const json = await res.json();
      
      const rawSyncs: any = {};
      const rawLogs: any = {};
      const datesSet = new Set<string>();

      json.syncs.forEach((s: any) => {
        rawSyncs[s.sync_date] = s;
        datesSet.add(s.sync_date);
      });
      json.logs.forEach((l: any) => {
        if (!rawLogs[l.log_date]) rawLogs[l.log_date] = [];
        rawLogs[l.log_date].push(l);
        datesSet.add(l.log_date);
      });

      const sortedDates = Array.from(datesSet).sort();
      const mergedList = sortedDates.map(date => {
        const dObj = new Date(date);
        const prevDate = new Date(dObj.getTime() - 86400000).toISOString().split('T')[0];
        
        const todaySync = rawSyncs[date] || {};
        const todayLogs = rawLogs[date] || [];
        const prevSync = rawSyncs[prevDate] || {};
        const prevLogs = rawLogs[prevDate] || [];

        const row: any = { date };

        // === OUTCOMES (Y) from Today ===
        row.hrv = todaySync.hrv_avg;
        row.rhr = todaySync.resting_hr;
        row.sleep = todaySync.sleep_score;
        row.stress_avg = todaySync.stress_avg;
        const weightLog = todayLogs.find((l: any) => l.log_type === 'weight');
        if (weightLog) row.weight = weightLog.value;

        // === INPUTS (X) from Yesterday (Lagged) ===
        row.active_calories = prevSync.active_calories;
        row.intensity_minutes = prevSync.intensity_minutes;
        
        // Extract high-fidelity Sleep Hours (Input) from TODAY'S raw payload
        // (Because Garmin reports sleep on the day you wake up)
        try {
          const raw = typeof todaySync.raw_payload === 'string' ? JSON.parse(todaySync.raw_payload) : todaySync.raw_payload;
          const sleepObj = raw?.sleep || {};
          const dto = sleepObj?.dailySleepDTO || {};
          const duration = sleepObj?.durationInSeconds || dto?.sleepDurationInSeconds || dto?.sleepTimeSeconds;
          if (duration) {
            row.sleep_h = (duration / 3600).toFixed(1);
          }
        } catch(e) {}

        // Macros & Quality from Yesterday's logs
        prevLogs.forEach((l: any) => {
          if (l.log_type === 'food') {
            row.calories = (row.calories || 0) + l.value;
            try {
              const parsed = typeof l.raw_input === 'string' ? JSON.parse(l.raw_input) : l.raw_input;
              if (parsed && parsed.parsed) {
                const p = parsed.parsed;
                row.protein = (row.protein || 0) + (p.protein_g || 0);
                row.carbs = (row.carbs || 0) + (p.carbs_g || 0);
                row.fat = (row.fat || 0) + (p.fat_g || 0);
                if (p.quality_score) row.quality = p.quality_score;
              }
            } catch(e) {}
          }
        });

        return row;
      });

      setHistory(mergedList.filter(r => r.hrv || r.weight || r.calories || r.active_calories));
    } catch (err) { console.error(err); }
  };

  const saveCreds = async () => {
    try {
      await fetch("/api/users/creds", {
        method: "POST",
        headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(creds)
      });
      alert("Credentials Encrypted & Saved! Garmin will now sync automatically.");
      setCreds({ email: "", password: "", name: creds.name });
    } catch (e) { console.error(e); }
  };

  const tabStyle = (tab: string) => ({
    padding: "0.5rem 1rem",
    background: activeTab === tab ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
    border: "none",
    color: activeTab === tab ? 'var(--accent-primary)' : 'var(--text-secondary)',
    cursor: "pointer" as const,
    fontWeight: 600,
    borderRadius: "var(--radius-sm)",
    transition: "all 0.2s ease",
  });

  return (
    <div className="premium-card" style={{ gridColumn: "1 / -1", transition: "all 0.5s ease" }}>
      {/* Header with Tabs */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "1rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>Wellness Orchestrator</h2>
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
            <button onClick={() => setActiveTab('dashboard')} style={tabStyle('dashboard')}>📊 Dashboard</button>
            <button onClick={() => setActiveTab('evals')} style={tabStyle('evals')}>🎯 Evals & Accuracy</button>
            <button onClick={() => setActiveTab('manual')} style={tabStyle('manual')}>📖 User Manual</button>
            <button onClick={() => setActiveTab('settings')} style={tabStyle('settings')}>⚙️ Settings</button>
            <button onClick={() => setActiveTab('admin')} style={{ ...tabStyle('admin'), background: activeTab === 'admin' ? 'rgba(245, 158, 11, 0.1)' : 'transparent', color: activeTab === 'admin' ? '#f59e0b' : 'var(--text-secondary)' }}>🔒 Admin</button>
          </div>
        </div>
        
        <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          {syncStatus && (
            <div style={{ fontSize: "0.8rem", color: syncStatus.message.startsWith("❌") ? "#ef4444" : "var(--text-secondary)" }}>
              {syncStatus.message}
            </div>
          )}
          <button 
            disabled={syncStatus?.syncing}
            onClick={async () => {
              setSyncStatus({ syncing: true, message: "⏳ Syncing with Garmin..." });
              try {
                const res = await fetch(`/api/garmin/sync?simulate=${simulateMode}`, {
                  headers: authHeaders
                });
                const data = await res.json();
                if (data.error) {
                  setSyncStatus({ syncing: false, message: `❌ Sync Failed: ${data.error}` });
                } else {
                  setSyncStatus({ syncing: false, message: `✅ Synced! (${data.source})` });
                  fetchHistory();
                }
              } catch (err) {
                setSyncStatus({ syncing: false, message: "❌ Network error" });
              }
            }}
            style={{ 
              background: "var(--accent-primary)", 
              color: "#fff", 
              border: "none", 
              padding: "0.5rem 1rem", 
              borderRadius: "var(--radius-sm)", 
              fontSize: "0.85rem", 
              fontWeight: 600, 
              cursor: syncStatus?.syncing ? "not-allowed" : "pointer" 
            }}
          >
            {syncStatus?.syncing ? "Syncing..." : "Sync Now"}
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Active Profile</span>
          <select 
            value={currentUserId} 
            onChange={(e) => setCurrentUserId(parseInt(e.target.value))}
            style={{ background: "var(--bg-card)", color: "#fff", border: "1px solid var(--border-color)", padding: "0.5rem", borderRadius: "var(--radius-sm)", outline: "none" }}
          >
            {users.map(u => <option key={u.id} value={u.id}>{u.username}</option>)}
          </select>
        </div>
      </div>

      {/* ============ DASHBOARD TAB ============ */}
      {activeTab === 'dashboard' && (
        <>
          {/* Goal Selector + Insight */}
          <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%)", border: "1px solid rgba(99, 102, 241, 0.2)", borderRadius: "var(--radius-md)" }}>
            {/* Goal Dropdown */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 600 }}>My Goal:</span>
              <select 
                value={selectedGoal} 
                onChange={(e) => setSelectedGoal(e.target.value)}
                style={{ 
                  background: "rgba(99, 102, 241, 0.15)", color: "var(--accent-primary)", 
                  border: "1px solid rgba(99, 102, 241, 0.3)", padding: "0.4rem 0.75rem", 
                  borderRadius: "var(--radius-sm)", outline: "none", fontWeight: 600, fontSize: "0.85rem",
                  cursor: "pointer"
                }}
              >
                {GOALS.map(g => <option key={g.value} value={g.value}>{g.label}</option>)}
              </select>

              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 600, marginLeft: "1rem" }}>Inference Mode:</span>
              <select 
                value={inferenceMode} 
                onChange={(e) => setInferenceMode(e.target.value)}
                style={{ 
                  background: "var(--bg)", color: "var(--text-primary)", 
                  border: "1px solid rgba(255, 255, 255, 0.1)", padding: "0.4rem 0.75rem", 
                  borderRadius: "var(--radius-sm)", outline: "none", fontSize: "0.85rem",
                  cursor: "pointer"
                }}
              >
                <option value="auto">🤖 Auto-Deploy (Recommended)</option>
                <option value="llm">🧠 Force LLM (Fidelity 1)</option>
                <option value="nn">⚡ Force NN (Fidelity 3)</option>
              </select>

              {insight?.fidelity_info && (
                <span style={{ 
                  fontSize: "0.65rem", padding: "0.15rem 0.5rem", borderRadius: "4px", fontWeight: 700,
                  background: `${insight.fidelity_info.color}20`, color: insight.fidelity_info.color,
                  textTransform: "uppercase", letterSpacing: "0.05em", marginLeft: "auto"
                }}>{insight.fidelity_info.label}</span>
              )}
            </div>
          </div>

          {/* AI Coach Synopsis */}
          {insight && (
            <div className="premium-card" style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "2rem", background: "linear-gradient(145deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.5) 100%)", border: "1px solid rgba(99, 102, 241, 0.2)", marginBottom: "2rem"}}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                  <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-primary)", boxShadow: "0 0 10px var(--accent-primary)"}}></div>
                  <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 600 }}>Causal AI Coach · {insight.goal.replace("_", " ")}</h3>
                </div>
                <p style={{ fontSize: "1rem", lineHeight: 1.6, color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
                  {insight.insight}
                </p>
                <div style={{ display: "flex", gap: "1.5rem" }}>
                  <div>
                    <div style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Prescribed Sleep</div>
                    <div style={{ fontWeight: 600 }}>{insight.recommendation.sleep}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Prescribed Exercise</div>
                    <div style={{ textTransform: "capitalize", fontWeight: 600 }}>{insight.recommendation.exercise.replace("_", " ")}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Prescribed Diet</div>
                    <div style={{ textTransform: "capitalize", fontWeight: 600 }}>{insight.recommendation.nutrition.replace("_", " ")}</div>
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: "1rem", paddingLeft: "2rem", borderLeft: "1px solid rgba(255,255,255,0.05)" }}>
                <div>
                  <div style={{ fontSize: "2rem", fontWeight: 700, color: insight.fidelity_info.color }}>Level {insight.fidelity}</div>
                  <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{insight.fidelity_info.label} Persona Engine</div>
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                  {insight.fidelity_info.desc}
                </div>
              </div>
            </div>
          )}
          {/* AI Coach Block moved to Evals Tab */}
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "1.5rem" }}>
            <h3 style={{ margin: 0 }}>Body State Summary</h3>
            {insight?.data_points && (
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                📊 {insight.data_points.syncs} syncs · {insight.data_points.logs} logs
              </span>
            )}
          </div>

          {/* Causal Lock Warning */}
          {(() => {
            const latest = history.length > 0 ? history[history.length - 1] : null;
            if (latest && latest.hrv && !latest.calories) {
              return (
                <div style={{ marginBottom: "1rem", padding: "1rem", background: "rgba(245, 158, 11, 0.1)", border: "1px solid var(--warning)", borderRadius: "var(--radius-sm)", color: "var(--warning)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "1.2rem" }}>⚠️</span>
                  <div>
                    <strong>Missing Dietary Inputs for Today!</strong> Action required: Please log your food below. The ML Regression Matrix cannot correctly map cause-and-effect without behavioral inputs.
                  </div>
                </div>
              );
            }
            return null;
          })()}

          {/* Historical Gap Scrubber */}
          {(() => {
            const gaps = history.filter(h => h.hrv && !h.calories && h.date !== new Date().toISOString().split('T')[0]);
            if (gaps.length === 0) return null;
            return (
              <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "rgba(99, 102, 241, 0.05)", border: "1px solid rgba(99, 102, 241, 0.2)", borderRadius: "var(--radius-md)" }}>
                <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  🔍 Regression Gap Analysis ({gaps.length} days missing logs)
                </h4>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {gaps.slice(-5).map(g => (
                    <button 
                      key={g.date}
                      onClick={() => {
                        setBackfillDate(g.date);
                        document.getElementById('manual-log-entry')?.scrollIntoView({ behavior: 'smooth' });
                      }}
                      style={{ 
                        background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", 
                        color: "#fff", padding: "0.4rem 0.8rem", borderRadius: "20px", fontSize: "0.75rem", 
                        cursor: "pointer", transition: "all 0.2s" 
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = "rgba(99, 102, 241, 0.2)"}
                      onMouseOut={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
                    >
                      🗓️ {g.date}
                    </button>
                  ))}
                  {gaps.length > 5 && <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", alignSelf: "center" }}>+{gaps.length - 5} more</span>}
                </div>
                <div style={{ marginTop: "0.75rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  Click a date to quickly backfill its nutritional data for the Digital Twin calibration.
                </div>
              </div>
            );
          })()}

          {/* Behavioral Inputs Grid (X) - 8 inputs */}
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <h4 style={{ margin: "0 0 1rem 0", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "0.8rem", letterSpacing: "0.05em" }}>Behavioral Inputs (The Causes)</h4>
              <span style={{ fontSize: "0.7rem", color: "var(--accent-primary)", fontWeight: 700 }}>Finalized Day T ({(() => {
                const latest = history.length > 0 ? history[history.length - 1] : null;
                if (!latest) return "—";
                const d = new Date(latest.date);
                return new Date(d.getTime() - 86400000).toISOString().split('T')[0];
              })()})</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "2rem" }}>
              {(() => {
                const latest = history.length > 0 ? history[history.length - 1] : null;
                return (
                  <>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #6366f1" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Sleep Duration</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.sleep_h ? `${latest.sleep_h}h` : "—"}</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #f59e0b" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Protein</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.protein?.toFixed(0) || "0"}g</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #10b981" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Carbohydrates</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.carbs?.toFixed(0) || "0"}g</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #f43f5e" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Fats</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.fat?.toFixed(0) || "0"}g</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #06b6d4" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Nutri Quality</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.quality || "—"}</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #ec4899" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Intensity Mins</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.intensity_minutes || "0"}m</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #8b5cf6" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Active Cals</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.active_calories || "0"}</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #64748b" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Dietary Cals</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.calories || "0"}</div>
                    </div>
                  </>
                )
              })()}
            </div>
          </div>

          {/* Biological Outcomes Grid (Y) - 5 outcomes */}
          <h4 style={{ margin: "0 0 1rem 0", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "0.8rem", letterSpacing: "0.05em" }}>Biological Outcomes (The Effects - Day T+1)</h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "1rem", marginBottom: "2rem" }}>
            {(() => {
              const latest = history.length > 0 ? history[history.length - 1] : null;
              return (
                <>
                  <div className="premium-card" style={{ padding: "1.25rem", textAlign: "center", borderTop: "3px solid var(--success)" }}>
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>HRV (ms)</div>
                     <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.5rem 0" }}>{latest?.hrv ?? "—"}</div>
                  </div>
                  <div className="premium-card" style={{ padding: "1.25rem", textAlign: "center", borderTop: "3px solid #ef4444" }}>
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>Resting HR</div>
                     <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.5rem 0" }}>{latest?.rhr || "—"}</div>
                  </div>
                  <div className="premium-card" style={{ padding: "1.25rem", textAlign: "center", borderTop: "3px solid #eab308" }}>
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>Garmin Stress</div>
                     <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.5rem 0" }}>{latest?.stress_avg || "—"}</div>
                  </div>
                  <div className="premium-card" style={{ padding: "1.25rem", textAlign: "center", borderTop: "3px solid #a855f7" }}>
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>Sleep Score</div>
                     <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.5rem 0" }}>{latest?.sleep || "—"}</div>
                  </div>
                  <div className="premium-card" style={{ padding: "1.25rem", textAlign: "center", borderTop: "3px solid var(--accent-primary)" }}>
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>Weight</div>
                     <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.5rem 0" }}>{latest?.weight ? `${latest.weight}kg` : "—"}</div>
                  </div>
                </>
              )
            })()}
          </div>
          
          {/* Causal Graphs Split */}
          <div style={{ display: "flex", flexDirection: "column", gap: "2rem", marginBottom: "2rem" }}>
            
            {/* Chart 1: Biological Outcomes (Y) */}
            <div className="premium-card">
              <h4 style={{ margin: "0 0 1rem 0", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "0.8rem", letterSpacing: "0.05em" }}>Biological Outcomes (Day T+1 Response)</h4>
              <div style={{ height: 250 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" fontSize={10} stroke="var(--text-secondary)" />
                    <YAxis fontSize={10} stroke="var(--text-secondary)" width={30} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'transparent' }} />
                    <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                    <Line type="monotone" dataKey="hrv" stroke="var(--success)" name="HRV" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="rhr" stroke="#ef4444" name="Resting HR" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="stress_avg" stroke="#eab308" name="Stress" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="sleep" stroke="#a855f7" name="Sleep" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="weight" stroke="var(--accent-primary)" name="Weight" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Behavioral Inputs (X) */}
            <div className="premium-card">
              <h4 style={{ margin: "0 0 1rem 0", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "0.8rem", letterSpacing: "0.05em" }}>Behavioral Inputs (Day T Actions)</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" fontSize={10} stroke="var(--text-secondary)" />
                    <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" width={30} label={{ value: 'g / h / Score', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)', fontSize: '0.7rem' } }} />
                    <YAxis yAxisId="right" orientation="right" fontSize={10} stroke="var(--accent-primary)" width={35} label={{ value: 'kcal', angle: 90, position: 'insideRight', style: { fill: 'var(--accent-primary)', fontSize: '0.7rem' } }} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'transparent' }} />
                    <Legend wrapperStyle={{ fontSize: "0.75rem", paddingTop: "1rem" }} />
                    
                    {/* Nutrient Macros (Left Axis) */}
                    <Line yAxisId="left" type="monotone" dataKey="protein" stroke="#f59e0b" name="Protein (g)" dot={false} strokeWidth={2} />
                    <Line yAxisId="left" type="monotone" dataKey="carbs" stroke="#10b981" name="Carbs (g)" dot={false} strokeWidth={2} />
                    <Line yAxisId="left" type="monotone" dataKey="fat" stroke="#f43f5e" name="Fat (g)" dot={false} strokeWidth={2} />
                    <Line yAxisId="left" type="monotone" dataKey="quality" stroke="#06b6d4" name="Quality" dot={false} strokeWidth={1} strokeDasharray="3 3" />
                    <Line yAxisId="left" type="monotone" dataKey="sleep_h" stroke="#6366f1" name="Sleep (h)" dot={false} strokeWidth={2} />
                    <Line yAxisId="left" type="monotone" dataKey="intensity_minutes" stroke="#ec4899" name="Intense Mins" dot={false} strokeWidth={2} />
                    
                    {/* Calories (Right Axis) */}
                    <Line yAxisId="right" type="monotone" dataKey="active_calories" stroke="#8b5cf6" name="Active Cals" dot={false} strokeWidth={3} />
                    <Line yAxisId="right" type="stepAfter" dataKey="calories" stroke="#64748b" name="Diet Cals" dot={false} strokeWidth={1} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
          </div>

          {/* Manual Log Form inline */}
          <div id="manual-log-entry" style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "1.5rem" }}>
            <ManualLogForm initialDate={backfillDate} />
          </div>
        </>
      )}
      {activeTab === 'evals' && (
        <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
          
          {/* AI Coach block removed from here */}

          {evals.length > 0 ? (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
              <div className="premium-card">
                <h3 style={{ margin: 0 }}>Expected vs Actual Outcomes (HRV)</h3>
                <div style={{ height: "300px", marginTop: "1rem" }}>
                  <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis fontSize={10} stroke="var(--text-secondary)" />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem", paddingTop: "10px" }} />
                        <Line type="monotone" dataKey="expected_hrv_delta" stroke="var(--text-secondary)" strokeWidth={2} name="Expected HRV Delta" dot={false} strokeDasharray="5 5"/>
                        <Line type="monotone" dataKey="actual_hrv_delta" stroke="var(--accent-primary)" strokeWidth={3} name="Actual HRV Delta" />
                      </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="premium-card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h3 style={{ margin: 0 }}>Model Accuracy & Fidelity</h3>
                  <div style={{ fontSize: "2rem", fontWeight: "700", color: "var(--status-good)"}}>
                    {(evals.reduce((acc, curr) => acc + (curr.fidelity_score || 0), 0) / evals.length * 100).toFixed(0)}%
                  </div>
                </div>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
                   Comparing LLM heuristic rules vs Neural Network probabilistic precision. Wait for NN deployment to see accuracy spike.
                </p>
                <div style={{ height: "250px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis domain={[0, 1]} tickFormatter={(i) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="var(--text-secondary)" />
                        <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem", paddingTop: "10px" }} />
                        <Line type="monotone" dataKey="fidelity_score" stroke="var(--accent-primary)" strokeWidth={3} name="Fidelity / Accuracy" dot={true} />
                      </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "4rem", background: "var(--bg-card)", borderRadius: "var(--radius-lg)" }}>
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📡</div>
              <h3 style={{ margin: "0 0 0.5rem 0" }}>Awaiting Lifecycle Synchronization</h3>
              <p style={{ color: "var(--text-secondary)" }}>
                Model evaluation mathematically requires 24-48 hours. We prescribe a habit today, wait for your metrics to sync tomorrow, and recursively score the delta. 
              </p>
            </div>
          )}
        </div>
      )}
      {/* ============ USER MANUAL TAB ============ */}
      {activeTab === 'manual' && <UserManual />}

      {/* ============ SETTINGS TAB ============ */}
      {activeTab === 'settings' && (
        <div style={{ padding: "1rem" }}>
          <h3>Profile Settings</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>Manage your profile and connect your Garmin account.</p>
          
          {/* --- Add New User --- */}
          <div style={{ marginTop: "1.5rem", padding: "1.25rem", background: "linear-gradient(135deg, rgba(34,197,94,0.05) 0%, rgba(34,197,94,0.02) 100%)", border: "1px solid rgba(34,197,94,0.2)", borderRadius: "var(--radius-md)" }}>
            <h4 style={{ margin: "0 0 0.75rem", color: "#22c55e" }}>➕ Add New User</h4>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", maxWidth: "400px" }}>
              <input type="text" placeholder="Username (unique)" value={creds.name || ""} onChange={(e) => setCreds({...creds, name: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
              <input type="email" placeholder="Garmin Email (optional)" value={creds.email} onChange={(e) => setCreds({...creds, email: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
              <input type="password" placeholder="Garmin Password (optional)" value={creds.password} onChange={(e) => setCreds({...creds, password: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
              <button onClick={async () => {
                if (!creds.name) { alert("Please enter a username"); return; }
                try {
                  const res = await fetch("/api/users/create", {
                    method: "POST",
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: creds.name, name: creds.name, email: creds.email, password: creds.password })
                  });
                  const result = await res.json();
                  if (result.exists) {
                    alert(`User "${creds.name}" already exists.`);
                    return;
                  }
                  
                  // Switch to the new user immediately
                  fetchUsers();
                  setCurrentUserId(result.id);
                  
                  // If Garmin creds were provided, trigger an immediate sync
                  if (creds.email && creds.password) {
                    setSyncStatus({ syncing: true, message: "⏳ Connecting to Garmin..." });
                    try {
                      const syncRes = await fetch("/api/garmin/sync", {
                        headers: { 'X-User-ID': result.id.toString() }
                      });
                      const syncData = await syncRes.json();
                      if (syncData.error) {
                        setSyncStatus({ syncing: false, message: `❌ Garmin connection failed: ${syncData.error}` });
                      } else if (syncData.source === "mock") {
                        setSyncStatus({ syncing: false, message: "⚠️ Could not authenticate with Garmin. Please verify your credentials." });
                      } else {
                        setSyncStatus({ syncing: false, message: `✅ Connected! Downloaded HRV: ${syncData.hrv?.lastNightAvg || '—'}ms, RHR: ${syncData.rhr?.restingHeartRate || '—'}bpm, Battery: ${syncData.body_battery?.latestValue || '—'}%` });
                        fetchHistory();
                      }
                    } catch (syncErr) {
                      setSyncStatus({ syncing: false, message: "❌ Network error during Garmin sync." });
                    }
                  } else {
                    setSyncStatus({ syncing: false, message: `✅ User "${creds.name}" created! Add Garmin credentials later to enable auto-sync.` });
                  }
                  
                  setCreds({ email: "", password: "", name: "" });
                } catch (e) { console.error(e); alert("Error creating user"); }
              }} style={{ background: "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)", color: "#fff", border: "none", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontWeight: 600, cursor: "pointer" }}>Create User & Connect Garmin</button>
            </div>
            {syncStatus && (
              <div style={{ marginTop: "1rem", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontSize: "0.85rem",
                background: syncStatus.message.startsWith("✅") ? "rgba(34,197,94,0.1)" : syncStatus.message.startsWith("❌") ? "rgba(239,68,68,0.1)" : "rgba(99,102,241,0.1)",
                border: `1px solid ${syncStatus.message.startsWith("✅") ? "rgba(34,197,94,0.3)" : syncStatus.message.startsWith("❌") ? "rgba(239,68,68,0.3)" : "rgba(99,102,241,0.3)"}`,
                color: syncStatus.message.startsWith("✅") ? "#22c55e" : syncStatus.message.startsWith("❌") ? "#ef4444" : "var(--accent-primary)"
              }}>
                {syncStatus.syncing && <span style={{ marginRight: "0.5rem" }}>⏳</span>}
                {syncStatus.message}
              </div>
            )}
          </div>
          
          {/* --- Update Existing User --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem", maxWidth: "400px", marginTop: "2rem" }}>
            <h4 style={{ margin: 0 }}>Update Current User: <span style={{ color: "var(--accent-primary)" }}>{users.find(u => u.id === currentUserId)?.username || "—"}</span></h4>
            
            <label style={{ fontSize: "0.8rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Garmin Connection</label>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", margin: 0 }}>Credentials are encrypted (AES-256). Once saved, Garmin data syncs automatically every 6 hours.</p>
            <input type="email" placeholder="Garmin Email" value={creds.email} onChange={(e) => setCreds({...creds, email: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
            <input type="password" placeholder="Garmin Password" value={creds.password} onChange={(e) => setCreds({...creds, password: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
            <button onClick={saveCreds} style={{ background: "var(--accent-primary)", color: "#fff", border: "none", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontWeight: 600, cursor: "pointer" }}>Save & Encrypt</button>
            
            <div style={{ marginTop: "2rem", padding: "1.25rem", background: "rgba(99,102,241,0.05)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: "var(--radius-md)" }}>
              <h4 style={{ margin: "0 0 0.5rem 0", color: "var(--accent-primary)" }}>🧪 Debug & Simulation</h4>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                Enable simulated data if your Garmin account is rate-limited (429) or if you want to test the causal matrix with high-fidelity mock values.
              </p>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <label className="switch">
                  <input type="checkbox" checked={simulateMode} onChange={(e) => setSimulatedMode(e.target.checked)} />
                  <span className="slider round"></span>
                </label>
                <span style={{ fontSize: "0.9rem", color: simulateMode ? "var(--accent-primary)" : "var(--text-secondary)", fontWeight: 600 }}>
                  {simulateMode ? "Simulated Mode ACTIVE" : "Simulated Mode DISABLED"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============ ADMIN TAB ============ */}
      {activeTab === 'admin' && <AdminPanel />}
    </div>
  );
}

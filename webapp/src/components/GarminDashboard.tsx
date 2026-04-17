"use client";

import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, AreaChart, Area, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
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

// ---- Apple Health Live Push Section (for Apple Watch users) ----
function AppleHealthPushSection({ userId }: { userId: number }) {
  const [open, setOpen] = React.useState(false);
  const apiBase = typeof window !== "undefined" ? `${window.location.origin}` : "";
  const endpointUrl = `${apiBase}/api/health/apple-push`;

  const exampleBody = JSON.stringify({
    date: new Date().toISOString().slice(0, 10),
    hrv: 45.2,
    resting_hr: 58,
    sleep_score: 82,
    sleep_hours: 7.5,
    steps: 8200,
    active_calories: 420,
    stress: 30,
  }, null, 2);

  return (
    <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "rgba(255,149,0,0.05)", border: "1px solid rgba(255,149,0,0.2)", borderRadius: "var(--radius-md)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }} onClick={() => setOpen(o => !o)}>
        <h4 style={{ margin: 0, color: "#ff9500" }}>📲 Apple Health Live Push (iOS Shortcut)</h4>
        <span style={{ color: "var(--text-secondary)", fontSize: "1.1rem" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ marginTop: "1rem" }}>
          <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: "0 0 1rem" }}>
            Instead of waiting for a scheduled Garmin sync, push your Apple Health data directly from iPhone using a free <strong style={{ color: "var(--text-primary)" }}>iOS Shortcut</strong>. Set it to run automatically at 9 PM every day.
          </p>

          {/* Step 1 */}
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "#ff9500", marginBottom: "0.4rem" }}>Step 1 — Create the Shortcut</div>
            <ol style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "1.2rem", margin: 0 }}>
              <li>Open <strong>Shortcuts</strong> app → + (new shortcut)</li>
              <li>Add: <strong>Get Health Samples</strong> — type: <em>Heart Rate Variability</em> (1 sample, most recent)</li>
              <li>Repeat for: Resting Heart Rate, Sleep Analysis, Step Count, Active Energy Burned, Heart Rate (stress proxy)</li>
              <li>Add: <strong>Get Contents of URL</strong></li>
              <li>Set URL to: <code style={{ fontSize: "0.75rem", background: "rgba(255,255,255,0.06)", padding: "0 4px", borderRadius: 3, wordBreak: "break-all" }}>{endpointUrl}</code></li>
              <li>Method: <strong>POST</strong> — Body: <strong>JSON</strong></li>
              <li>Add fields: date, hrv, resting_hr, sleep_hours, steps, active_calories (map each from the Health Sample variables above)</li>
              <li>Add header: <code style={{ fontSize: "0.75rem", background: "rgba(255,255,255,0.06)", padding: "0 4px", borderRadius: 3 }}>x-user-id</code> = <strong>{userId}</strong></li>
            </ol>
          </div>

          {/* Step 2 */}
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "#ff9500", marginBottom: "0.4rem" }}>Step 2 — Automate at 9 PM</div>
            <ol style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "1.2rem", margin: 0 }}>
              <li>Tap <strong>Automation</strong> tab → + → <em>Time of Day</em></li>
              <li>Set time to <strong>9:00 PM</strong>, daily</li>
              <li>Run the shortcut — enable <strong>Run Immediately</strong> (no confirmation pop-up)</li>
            </ol>
          </div>

          {/* JSON example */}
          <div>
            <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "#ff9500", marginBottom: "0.4rem" }}>JSON payload format</div>
            <pre style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "var(--radius-sm)", padding: "0.75rem", fontSize: "0.72rem", color: "#9ca3af", overflowX: "auto", margin: 0 }}>
              {exampleBody}
            </pre>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: "0.5rem 0 0" }}>
              All fields are optional — send whichever metrics Apple Health has for today.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

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

  const [creds, setCreds] = useState({ email: "", password: "" });
  const [userDevice, setUserDevice] = useState<string>("garmin");
  const [uploadStatus, setUploadStatus] = useState<{uploading: boolean, message: string} | null>(null);
  const [newUserDevice, setNewUserDevice] = useState<string>("");
  const [newUserCreds, setNewUserCreds] = useState({ name: "", email: "", password: "" });

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
        const active = defaultUser || list[0];
        setCurrentUserId(active.id);
        setUserDevice(active.wearable_source || "garmin");
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
        row.hrv = todaySync.hrv_rmssd;
        row.rhr = todaySync.resting_hr;
        row.sleep = todaySync.sleep_score;
        row.stress_avg = todaySync.stress_avg;
        const weightLog = todayLogs.find((l: any) => l.log_type === 'weight');
        if (weightLog) row.weight = weightLog.value;

        // === INPUTS (X) from Yesterday (Lagged) ===
        row.active_calories = prevSync.active_calories;
        row.active_minutes = prevSync.active_minutes;
        
        // Also store today's actual wearable values for verification (shown in summary cards)
        row.today_active_minutes = todaySync.active_minutes;
        row.today_active_calories = todaySync.active_calories;
        row.today_steps = todaySync.steps;
        
        // Sleep duration: use stored column first, fall back to raw payload
        if (todaySync.sleep_duration_hours) {
          row.sleep_h = todaySync.sleep_duration_hours;
        } else {
          try {
            const raw = typeof todaySync.raw_payload === 'string' ? JSON.parse(todaySync.raw_payload) : todaySync.raw_payload;
            const sleepObj = raw?.sleep || {};
            const dto = sleepObj?.dailySleepDTO || {};
            const duration = sleepObj?.durationInSeconds || dto?.sleepDurationInSeconds || dto?.sleepTimeSeconds;
            if (duration) row.sleep_h = parseFloat((duration / 3600).toFixed(1));
          } catch(e) {}
        }

        // Macros & Quality from Yesterday's logs
        const nutritionTypeToScore: Record<string, number> = {
          high_protein: 8, balanced: 7, high_carb: 5, processed: 3, skipped: 0
        };
        let qualityScores: number[] = [];
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
                const qs = p.quality_score ?? nutritionTypeToScore[p.nutrition_type] ?? null;
                if (qs !== null) qualityScores.push(qs);
              }
            } catch(e) {}
          }
        });
        if (qualityScores.length > 0) {
          row.quality = parseFloat((qualityScores.reduce((a, b) => a + b, 0) / qualityScores.length).toFixed(1));
        }

        return row;
      });

      const filtered = mergedList.filter(r => r.hrv || r.rhr || r.sleep || r.stress_avg || r.weight || r.calories || r.active_calories || r.sleep_h);
      console.log('[WE] history merged:', mergedList.length, 'filtered:', filtered.length, JSON.stringify(filtered.map(r => ({d: r.date, hrv: r.hrv, rhr: r.rhr, sleep: r.sleep, stress: r.stress_avg, sleep_h: r.sleep_h}))));
      setHistory(filtered);
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
      setCreds({ email: "", password: "" });
    } catch (e) { console.error(e); }
  };

  const tabStyle = (tab: string) => ({
    padding: "0.5rem 0.75rem",
    background: activeTab === tab ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
    border: "none",
    color: activeTab === tab ? 'var(--accent-primary)' : 'var(--text-secondary)',
    cursor: "pointer" as const,
    fontWeight: 600,
    borderRadius: "var(--radius-sm)",
    transition: "all 0.2s ease",
    fontSize: "0.85rem",
    whiteSpace: "nowrap" as const,
  });

  return (
    <div className="premium-card" style={{ gridColumn: "1 / -1", transition: "all 0.5s ease" }}>
      {/* Header with Tabs */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginBottom: "1.5rem", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "1rem" }}>
        {/* Title + Controls Row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
          <h2 style={{ fontSize: "clamp(1.1rem, 3vw, 1.5rem)", fontWeight: 700, margin: 0 }}>Wellness Orchestrator</h2>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            {syncStatus && (
              <div style={{ fontSize: "0.8rem", color: syncStatus.message.startsWith("❌") ? "#ef4444" : syncStatus.message.startsWith("⚠️") ? "#f59e0b" : "var(--text-secondary)" }}>
                {syncStatus.message}
              </div>
            )}
          <button 
            disabled={syncStatus?.syncing || uploadStatus?.uploading}
            onClick={async () => {
              if (userDevice === "garmin") {
                setSyncStatus({ syncing: true, message: "⏳ Syncing with Garmin..." });
                try {
                  const res = await fetch(`/api/garmin/sync?simulate=${simulateMode}`, { headers: authHeaders });
                  const data = await res.json();
                  const results: any[] = data.results || [];
                  const rateLimited = results.some((r: any) => r.status === "rate_limited");
                  const anySuccess = results.some((r: any) => r.status === "success");
                  const backoff = data.backoff;
                  if (rateLimited) {
                    const retryMsg = backoff?.next_retry_after 
                      ? ` Retry #${backoff.failures} in ~${Math.ceil((new Date(backoff.next_retry_after).getTime() - Date.now()) / 60000)}min.`
                      : "";
                    setSyncStatus({ syncing: false, message: `⚠️ Garmin rate limited — using stored data.${retryMsg}` });
                    fetchHistory();
                    fetchUsers();
                  } else if (data.error || results.every((r: any) => r.status === "error")) {
                    const errMsg = results.find((r: any) => r.message)?.message || data.error || "Unknown error";
                    setSyncStatus({ syncing: false, message: `❌ Sync failed: ${errMsg}` });
                  } else {
                    setSyncStatus({ syncing: false, message: `✅ Synced! (${results.find((r: any) => r.source)?.source || "garmin"})` });
                    fetchHistory();
                    fetchUsers();
                  }
                } catch (err) {
                  setSyncStatus({ syncing: false, message: "❌ Network error" });
                }
              } else {
                setActiveTab('settings');
              }
            }}
            style={{ 
              background: userDevice === "garmin" ? "var(--accent-primary)" : "rgba(99,102,241,0.2)",
              color: "#fff", border: userDevice === "garmin" ? "none" : "1px solid rgba(99,102,241,0.4)",
              padding: "0.5rem 1rem", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", fontWeight: 600,
              cursor: (syncStatus?.syncing || uploadStatus?.uploading) ? "not-allowed" : "pointer"
            }}
          >
            {syncStatus?.syncing ? "Syncing..." : userDevice === "garmin" ? "Sync Now" : "📤 Upload Data"}
          </button>
          </div>
        </div>

        {/* Profile Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Active Profile</span>
          <select 
            value={currentUserId} 
            onChange={(e) => {
              const uid = parseInt(e.target.value);
              setCurrentUserId(uid);
              const u = users.find((u: any) => u.id === uid);
              if (u) setUserDevice(u.wearable_source || "garmin");
            }}
            style={{ background: "var(--bg-card)", color: "#fff", border: "1px solid var(--border-color)", padding: "0.5rem", borderRadius: "var(--radius-sm)", outline: "none" }}
          >
            {users.map(u => <option key={u.id} value={u.id}>{u.username}</option>)}
          </select>
          {(() => {
            const u = users.find((u: any) => u.id === currentUserId);
            if (!u?.last_synced_at) return (
              <span style={{ fontSize: "0.65rem", color: "#ef4444" }}>⚠ Never synced</span>
            );
            const synced = new Date(u.last_synced_at);
            const hoursAgo = (Date.now() - synced.getTime()) / 3600000;
            const color = hoursAgo < 13 ? "#22c55e" : hoursAgo < 25 ? "#f59e0b" : "#ef4444";
            const label = hoursAgo < 1 ? "< 1h ago"
              : hoursAgo < 24 ? `${Math.floor(hoursAgo)}h ago`
              : `${Math.floor(hoursAgo / 24)}d ago`;
            const icon = hoursAgo < 13 ? "✅" : hoursAgo < 25 ? "⚠️" : "❌";
            return (
              <span style={{ fontSize: "0.65rem", color }} title={`Last synced: ${synced.toLocaleString()}\nLast data: ${u.last_sync_date || "—"}`}>
                {icon} Synced {label} · data thru {u.last_sync_date || "—"}
              </span>
            );
          })()}
        </div>
        {/* Tab Navigation */}
        <div style={{ display: "flex", gap: "0.25rem", overflowX: "auto", WebkitOverflowScrolling: "touch", scrollbarWidth: "none", msOverflowStyle: "none" }}>
            <button onClick={() => setActiveTab('dashboard')} style={tabStyle('dashboard')}>📊 Dashboard</button>
            <button onClick={() => setActiveTab('evals')} style={tabStyle('evals')}>🎯 Evals</button>
            <button onClick={() => setActiveTab('manual')} style={tabStyle('manual')}>📖 Manual</button>
            <button onClick={() => setActiveTab('settings')} style={tabStyle('settings')}>⚙️ Settings</button>
            <button onClick={() => setActiveTab('admin')} style={{ ...tabStyle('admin'), background: activeTab === 'admin' ? 'rgba(245, 158, 11, 0.1)' : 'transparent', color: activeTab === 'admin' ? '#f59e0b' : 'var(--text-secondary)' }}>🔒 Admin</button>
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
            <div className="premium-card" style={{ display: "flex", flexDirection: "column", gap: "1.5rem", background: "linear-gradient(145deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.5) 100%)", border: "1px solid rgba(99, 102, 241, 0.2)", marginBottom: "2rem"}}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "1.5rem" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                    <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-primary)", boxShadow: "0 0 10px var(--accent-primary)"}}></div>
                    <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 600 }}>Causal AI Coach · {insight.goal.replace("_", " ")}</h3>
                  </div>
                  <p style={{ fontSize: "1rem", lineHeight: 1.6, color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
                    {insight.insight}
                  </p>
                  <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
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
                <div style={{ display: "flex", flexDirection: "row", justifyContent: "flex-start", gap: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.05)", flexWrap: "wrap" }}>
                  <div>
                    <div style={{ fontSize: "2rem", fontWeight: 700, color: insight.fidelity_info.color }}>Level {insight.fidelity}</div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{insight.fidelity_info.label} Persona Engine</div>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                    {insight.fidelity_info.desc}
                  </div>
                </div>
              </div>

              {/* Expected Realistic Outcomes */}
              {insight.expected_deltas && (
                <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "1.25rem" }}>
                  <h4 style={{ margin: "0 0 1rem 0", fontSize: "0.85rem", color: "var(--accent-primary)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 700 }}>Expected Realistic Outcomes (Next Day)</h4>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: "0.75rem" }}>
                    {[
                      { key: "hrv", label: "HRV", unit: "ms", better: "up" },
                      { key: "resting_hr", label: "Resting HR", unit: "bpm", better: "down" },
                      { key: "sleep_efficiency", label: "Sleep Score", unit: "pts", better: "up" },
                      { key: "cortisol_proxy", label: "Stress", unit: "pts", better: "down" },
                      { key: "body_fat_pct", label: "Body Fat", unit: "%", better: "down" },
                      { key: "energy_level", label: "Energy", unit: "pts", better: "up" },
                    ].map(({ key, label, unit, better }) => {
                      const val = insight.expected_deltas[key];
                      if (val == null) return null;
                      const isImproving = better === "up" ? val > 0 : val < 0;
                      const isNeutral = Math.abs(val) < 0.01;
                      const color = isNeutral ? "var(--text-secondary)" : isImproving ? "#22c55e" : "#f59e0b";
                      const arrow = isNeutral ? "→" : val > 0 ? "↑" : "↓";
                      return (
                        <div key={key} style={{ 
                          padding: "0.75rem", borderRadius: "var(--radius-sm)", textAlign: "center",
                          background: `${color}10`, border: `1px solid ${color}30`
                        }}>
                          <div style={{ fontSize: "0.65rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "0.25rem" }}>{label}</div>
                          <div style={{ fontSize: "1.25rem", fontWeight: 800, color }}>
                            {val > 0 ? "+" : ""}{Math.abs(val) < 0.01 ? val.toFixed(3) : val.toFixed(1)}{unit !== "%" ? "" : "%"}
                          </div>
                          <div style={{ fontSize: "0.75rem", color, fontWeight: 600 }}>{arrow} {unit}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Long-Term Healthspan / Lifespan Impact */}
              {insight.long_term_impact && (
                <div style={{ 
                  borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "1.25rem",
                  background: "linear-gradient(135deg, rgba(34,197,94,0.05) 0%, rgba(99,102,241,0.05) 100%)",
                  borderRadius: "var(--radius-sm)", padding: "1.25rem", margin: "0 -1.5rem -1.5rem -1.5rem",
                  borderBottomLeftRadius: "var(--radius-lg)", borderBottomRightRadius: "var(--radius-lg)"
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                    <span style={{ fontSize: "1.25rem" }}>🧬</span>
                    <h4 style={{ margin: 0, fontSize: "0.85rem", color: "#22c55e", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 700 }}>Long-Term Healthspan & Lifespan Impact</h4>
                  </div>
                  <p style={{ fontSize: "0.9rem", lineHeight: 1.7, color: "var(--text-secondary)", margin: 0 }}>
                    {insight.long_term_impact}
                  </p>
                </div>
              )}
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
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
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
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.active_minutes || latest?.today_active_minutes || "0"}m</div>
                       <div style={{ fontSize: "0.6rem", color: "var(--text-secondary)" }}>yesterday → today</div>
                    </div>
                    <div className="premium-card" style={{ padding: "1rem", textAlign: "center", borderTop: "3px solid #8b5cf6" }}>
                       <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Active Cals</div>
                       <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)", margin: "0.25rem 0" }}>{latest?.active_calories || latest?.today_active_calories || "0"}</div>
                       <div style={{ fontSize: "0.6rem", color: "var(--text-secondary)" }}>yesterday → today</div>
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
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
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
                     <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>Stress</div>
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

            {/* Chart 2: Behavioral Inputs — small multiple sparklines */}
            <div className="premium-card">
              <h4 style={{ margin: "0 0 1.25rem 0", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "0.8rem", letterSpacing: "0.05em" }}>Behavioral Inputs (Day T Actions)</h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "1rem" }}>
                {[
                  { key: "sleep_h",        label: "Sleep",        unit: "h",    color: "#6366f1" },
                  { key: "protein",        label: "Protein",      unit: "g",    color: "#f59e0b" },
                  { key: "carbs",          label: "Carbs",        unit: "g",    color: "#10b981" },
                  { key: "fat",            label: "Fat",          unit: "g",    color: "#f43f5e" },
                  { key: "quality",        label: "Nutri Quality",unit: "/10",  color: "#06b6d4" },
                  { key: "calories",       label: "Diet Cals",    unit: "kcal", color: "#64748b" },
                  { key: "active_minutes", label: "Active Mins",  unit: "min",  color: "#ec4899" },
                  { key: "active_calories",label: "Active Cals",  unit: "kcal", color: "#8b5cf6" },
                ].map(({ key, label, unit, color }) => {
                  const latest = history.length > 0 ? history[history.length - 1]?.[key] : null;
                  return (
                    <div key={key} style={{ background: "rgba(255,255,255,0.03)", borderRadius: "var(--radius-sm)", padding: "0.75rem", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.5rem" }}>
                        <span style={{ fontSize: "0.65rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 600 }}>{label}</span>
                        <span style={{ fontSize: "0.9rem", fontWeight: 700, color }}>
                          {latest != null ? `${typeof latest === "number" ? latest.toFixed(key === "quality" ? 1 : 0) : latest}${unit}` : "—"}
                        </span>
                      </div>
                      <div style={{ height: 60 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={history} margin={{ top: 4, right: 2, left: 2, bottom: 0 }}>
                            <YAxis domain={["auto", "auto"]} hide />
                            <Tooltip
                              contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.75rem", borderRadius: "8px", padding: "4px 8px" }}
                              formatter={(v: any) => [`${typeof v === "number" ? v.toFixed(key === "quality" ? 1 : 0) : v}${unit}`, label]}
                              labelFormatter={(l) => l}
                            />
                            <Line type="monotone" dataKey={key} stroke={color} dot={false} strokeWidth={2} connectNulls />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
          </div>

          {/* Manual Log Form inline */}
          <div id="manual-log-entry" style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "1.5rem" }}>
            <ManualLogForm initialDate={backfillDate} userId={currentUserId} onSaved={fetchHistory} />
          </div>
        </>
      )}
      {activeTab === 'evals' && (
        <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
          
          {/* AI Coach block removed from here */}

          {evals.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "2.5rem" }}>

              {/* ====== Summary Fidelity & Overall Compliance ====== */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
                <div className="premium-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3 style={{ margin: 0 }}>Model Accuracy & Fidelity</h3>
                    <div style={{ fontSize: "2rem", fontWeight: "700", color: "var(--status-good)"}}>
                      {(evals.reduce((acc, curr) => acc + (curr.fidelity_score || 0), 0) / evals.length * 100).toFixed(0)}%
                    </div>
                  </div>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                    Overall prediction accuracy across all metrics. Higher fidelity means the model's expected outcomes closely match your actual results.
                  </p>
                  <div style={{ height: "200px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="var(--text-secondary)" />
                        <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem", paddingTop: "10px" }} />
                        <Line type="monotone" dataKey="fidelity_score" stroke="var(--accent-primary)" strokeWidth={3} name="Fidelity / Accuracy" dot={true} />
                        <Line type="monotone" dataKey="compliance_score" stroke="#22c55e" strokeWidth={2} name="Overall Compliance" dot={false} strokeDasharray="5 5" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="premium-card">
                  <h3 style={{ margin: "0 0 0.5rem" }}>Per-Input Compliance Breakdown</h3>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                    How closely you followed each type of recommendation. High compliance correlates with better prediction accuracy.
                  </p>
                  <div style={{ height: "200px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="var(--text-secondary)" />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem", paddingTop: "10px" }} />
                        <Bar dataKey="compliance_sleep" fill="#6366f1" name="Sleep Compliance" />
                        <Bar dataKey="compliance_exercise" fill="#ec4899" name="Exercise Compliance" />
                        <Bar dataKey="compliance_nutrition" fill="#f59e0b" name="Nutrition Compliance" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* ====== METRIC 1: HRV — Outcome vs Sleep Compliance ====== */}
              <div className="premium-card">
                <h3 style={{ margin: "0 0 0.25rem" }}>HRV (Heart Rate Variability)</h3>
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" label={{ value: 'Delta', angle: -90, position: 'insideLeft', style: { fontSize: '0.7rem', fill: 'var(--text-secondary)' } }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="#6366f1" label={{ value: 'Compliance %', angle: 90, position: 'insideRight', style: { fontSize: '0.7rem', fill: '#6366f1' } }} />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} formatter={(v: number, name: string) => name.includes('Compliance') ? [(v*100).toFixed(0) + '%', name] : [v?.toFixed?.(2) ?? v, name]} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_sleep" stroke="#6366f1" fill="rgba(99,102,241,0.12)" strokeWidth={1.5} name="Sleep Compliance %" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="expected_hrv_delta" stroke="#9ca3af" strokeWidth={2} name="Expected HRV" dot={{ r: 4 }} strokeDasharray="5 5" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="actual_hrv_delta" stroke="#22c55e" strokeWidth={3} name="Actual HRV" dot={{ r: 4 }} connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>HRV</strong> measures your autonomic nervous system's recovery capacity. Sleep is the primary driver of HRV improvement — when sleep compliance is high (you sleep within the recommended duration), HRV outcomes tend to closely match predictions. Gaps between expected and actual HRV often correlate with nights of insufficient or excessive sleep, late alcohol consumption, or elevated pre-sleep stress.
                  </p>
                </div>
              </div>

              {/* ====== METRIC 2: RHR — Outcome vs Exercise Compliance ====== */}
              <div className="premium-card">
                <h3 style={{ margin: "0 0 0.25rem" }}>Resting Heart Rate (RHR)</h3>
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" label={{ value: 'Delta', angle: -90, position: 'insideLeft', style: { fontSize: '0.7rem', fill: 'var(--text-secondary)' } }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="#ec4899" label={{ value: 'Compliance %', angle: 90, position: 'insideRight', style: { fontSize: '0.7rem', fill: '#ec4899' } }} />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} formatter={(v: number, name: string) => name.includes('Compliance') ? [(v*100).toFixed(0) + '%', name] : [v?.toFixed?.(2) ?? v, name]} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_exercise" stroke="#ec4899" fill="rgba(236,72,153,0.12)" strokeWidth={1.5} name="Exercise Compliance %" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="expected_rhr_delta" stroke="#9ca3af" strokeWidth={2} name="Expected RHR" dot={{ r: 4 }} strokeDasharray="5 5" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="actual_rhr_delta" stroke="#ef4444" strokeWidth={3} name="Actual RHR" dot={{ r: 4 }} connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>Resting Heart Rate</strong> reflects cardiovascular fitness and recovery status. Exercise compliance is the primary lever — consistent moderate exercise progressively lowers RHR over weeks. When exercise recommendations are followed, RHR predictions align closely with actuals. Missed workouts create upward drift, while overtraining without recovery days causes acute spikes.
                  </p>
                </div>
              </div>

              {/* ====== METRIC 3: Sleep Score — Outcome vs Sleep Compliance ====== */}
              <div className="premium-card">
                <h3 style={{ margin: "0 0 0.25rem" }}>Sleep Score</h3>
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" label={{ value: 'Delta', angle: -90, position: 'insideLeft', style: { fontSize: '0.7rem', fill: 'var(--text-secondary)' } }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="#6366f1" label={{ value: 'Compliance %', angle: 90, position: 'insideRight', style: { fontSize: '0.7rem', fill: '#6366f1' } }} />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} formatter={(v: number, name: string) => name.includes('Compliance') ? [(v*100).toFixed(0) + '%', name] : [v?.toFixed?.(2) ?? v, name]} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_sleep" stroke="#6366f1" fill="rgba(99,102,241,0.12)" strokeWidth={1.5} name="Sleep Compliance %" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="expected_sleep_delta" stroke="#9ca3af" strokeWidth={2} name="Expected Sleep" dot={{ r: 4 }} strokeDasharray="5 5" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="actual_sleep_delta" stroke="#a855f7" strokeWidth={3} name="Actual Sleep" dot={{ r: 4 }} connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>Sleep Score</strong> reflects overall sleep quality including duration, depth, and consistency. Sleep duration compliance directly impacts this score — sleeping within the recommended window improves deep sleep phases and next-day recovery. Inconsistent sleep timing, even with adequate duration, can reduce the score due to circadian rhythm disruption.
                  </p>
                </div>
              </div>

              {/* ====== METRIC 4: Stress — Outcome vs Combined Compliance ====== */}
              <div className="premium-card">
                <h3 style={{ margin: "0 0 0.25rem" }}>Stress Level (Cortisol Proxy)</h3>
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" label={{ value: 'Delta', angle: -90, position: 'insideLeft', style: { fontSize: '0.7rem', fill: 'var(--text-secondary)' } }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="#8b5cf6" label={{ value: 'Compliance %', angle: 90, position: 'insideRight', style: { fontSize: '0.7rem', fill: '#8b5cf6' } }} />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} formatter={(v: number, name: string) => name.includes('Compliance') ? [(v*100).toFixed(0) + '%', name] : [v?.toFixed?.(2) ?? v, name]} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_sleep" stroke="#6366f1" fill="rgba(99,102,241,0.10)" strokeWidth={1.5} name="Sleep Compliance %" connectNulls />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_exercise" stroke="#ec4899" fill="rgba(236,72,153,0.10)" strokeWidth={1.5} name="Exercise Compliance %" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="expected_stress_delta" stroke="#9ca3af" strokeWidth={2} name="Expected Stress" dot={{ r: 4 }} strokeDasharray="5 5" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="actual_stress_delta" stroke="#eab308" strokeWidth={3} name="Actual Stress" dot={{ r: 4 }} connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>Stress levels</strong> (cortisol proxy) are influenced by both sleep quality and exercise type. Yoga and light cardio reduce stress, while HIIT temporarily elevates it before recovery. High sleep compliance combined with appropriate exercise compliance produces the best stress reduction outcomes. Nutrition quality also plays a moderating role — processed foods and skipped meals elevate cortisol.
                  </p>
                </div>
              </div>

              {/* ====== METRIC 5: Weight / Body Composition — Outcome vs Nutrition Compliance ====== */}
              <div className="premium-card">
                <h3 style={{ margin: "0 0 0.25rem" }}>Body Composition / Weight</h3>
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={evals.slice(-14)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="rec_date" fontSize={10} stroke="var(--text-secondary)" />
                        <YAxis yAxisId="left" fontSize={10} stroke="var(--text-secondary)" label={{ value: 'Delta', angle: -90, position: 'insideLeft', style: { fontSize: '0.7rem', fill: 'var(--text-secondary)' } }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(i: number) => (i*100).toFixed(0) + '%'} fontSize={10} stroke="#f59e0b" label={{ value: 'Compliance %', angle: 90, position: 'insideRight', style: { fontSize: '0.7rem', fill: '#f59e0b' } }} />
                        <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--border-color)", fontSize: "0.85rem", borderRadius: "10px" }} formatter={(v: number, name: string) => name.includes('Compliance') ? [(v*100).toFixed(0) + '%', name] : [v?.toFixed?.(2) ?? v, name]} />
                        <Legend wrapperStyle={{ fontSize: "0.85rem" }} />
                        <Area yAxisId="right" type="monotone" dataKey="compliance_nutrition" stroke="#f59e0b" fill="rgba(245,158,11,0.12)" strokeWidth={1.5} name="Nutrition Compliance %" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="expected_weight_delta" stroke="#9ca3af" strokeWidth={2} name="Expected (Body Fat %)" dot={{ r: 4 }} strokeDasharray="5 5" connectNulls />
                        <Line yAxisId="left" type="monotone" dataKey="actual_weight_delta" stroke="var(--accent-primary)" strokeWidth={3} name="Actual" dot={{ r: 4 }} connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>Body composition</strong> changes are primarily driven by nutrition compliance. High-protein diets combined with strength training produce the best fat loss and lean mass preservation. This metric moves slowly — expect meaningful changes over 2-4 weeks of consistent compliance rather than daily fluctuations. Weight logging via manual entries is required for actual tracking.
                  </p>
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
        <div style={{ padding: "1rem", maxWidth: 680 }}>
          <h3 style={{ margin: "0 0 0.25rem" }}>Settings</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "2rem" }}>Manage your profile, device, and health data.</p>

          {/* ---- 1. Add New User (top) ---- */}
          <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "rgba(34,197,94,0.04)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: "var(--radius-md)" }}>
            <h4 style={{ margin: "0 0 0.5rem", color: "#22c55e" }}>➕ Add New User</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "0 0 0.75rem" }}>Create a profile, then select a device and upload data below.</p>
            <div style={{ display: "flex", gap: "0.75rem", maxWidth: 400 }}>
              <input type="text" placeholder="Username (unique)" value={newUserCreds.name} onChange={e => setNewUserCreds({...newUserCreds, name: e.target.value})} style={{ flex: 1, background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
              <button disabled={!newUserCreds.name} onClick={async () => {
                if (!newUserCreds.name) return;
                try {
                  const res = await fetch("/api/users/create", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username: newUserCreds.name, name: newUserCreds.name, device: "garmin" })
                  });
                  const result = await res.json();
                  if (result.exists) { alert(`User "${newUserCreds.name}" already exists.`); return; }
                  fetchUsers();
                  setCurrentUserId(result.id);
                  setUserDevice("garmin");
                  setNewUserCreds({ name: "", email: "", password: "" });
                  setSyncStatus({ syncing: false, message: `✅ User "${newUserCreds.name}" created! Select a device and configure below.` });
                } catch { alert("Error creating user"); }
              }} style={{ background: newUserCreds.name ? "#22c55e" : "rgba(34,197,94,0.3)", color: "#fff", border: "none", padding: "0.75rem 1.25rem", borderRadius: "var(--radius-sm)", fontWeight: 600, cursor: newUserCreds.name ? "pointer" : "not-allowed", whiteSpace: "nowrap" }}>
                Create
              </button>
            </div>
            {syncStatus && <div style={{ marginTop: "0.75rem", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", background: syncStatus.message.startsWith("✅") ? "rgba(34,197,94,0.1)" : "rgba(99,102,241,0.1)", color: syncStatus.message.startsWith("✅") ? "#22c55e" : "var(--accent-primary)" }}>{syncStatus.message}</div>}
          </div>

          {/* ---- 2. Active Profile: Device Selection ---- */}
          <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "rgba(99,102,241,0.05)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: "var(--radius-md)" }}>
            <h4 style={{ margin: "0 0 0.25rem", color: "var(--accent-primary)" }}>📱 Active Profile Device</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "0 0 1rem" }}>Change the wearable type for the currently selected user.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
              {[
                { id: "garmin", label: "Garmin", icon: "⌚" },
                { id: "apple_watch", label: "Apple Watch", icon: "🍎" },
                { id: "oneplus", label: "OnePlus Watch", icon: "📱" },
                { id: "other", label: "Other / CSV", icon: "📊" },
              ].map(d => (
                <button key={d.id} onClick={async () => {
                  await fetch("/api/users/device", {
                    method: "POST",
                    headers: { ...authHeaders, "Content-Type": "application/json" },
                    body: JSON.stringify({ device: d.id })
                  });
                  setUserDevice(d.id);
                  setUsers(users.map((u: any) => u.id === currentUserId ? { ...u, wearable_source: d.id } : u));
                }} style={{
                  padding: "1rem 0.5rem", borderRadius: "var(--radius-sm)", textAlign: "center",
                  background: userDevice === d.id ? "rgba(99,102,241,0.25)" : "var(--bg-card)",
                  border: userDevice === d.id ? "2px solid var(--accent-primary)" : "1px solid var(--border-color)",
                  color: userDevice === d.id ? "var(--accent-primary)" : "var(--text-secondary)",
                  cursor: "pointer", fontWeight: 600, fontSize: "0.85rem", transition: "all 0.15s"
                }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: "0.35rem" }}>{d.icon}</div>
                  {d.label}
                  {userDevice === d.id && <div style={{ fontSize: "0.7rem", marginTop: "0.25rem" }}>✓ Active</div>}
                </button>
              ))}
            </div>

            {/* Garmin Credentials inline */}
            {userDevice === "garmin" && (
              <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "0 0 0.75rem" }}>🔗 Garmin credentials are encrypted (AES-256). Data syncs at 9 AM & 9 PM daily.</p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", maxWidth: 400 }}>
                  <input type="email" placeholder="Garmin Email" value={creds.email} onChange={e => setCreds({...creds, email: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
                  <input type="password" placeholder="Garmin Password" value={creds.password} onChange={e => setCreds({...creds, password: e.target.value})} style={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", padding: "0.75rem", borderRadius: "var(--radius-sm)", color: "#fff" }} />
                  <button onClick={saveCreds} style={{ background: "#22c55e", color: "#fff", border: "none", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontWeight: 600, cursor: "pointer" }}>Save & Encrypt</button>
                </div>
              </div>
            )}
          </div>

          {/* ---- 3. Upload Health Data (always available for active profile) ---- */}
          <div style={{ marginBottom: "2rem", padding: "1.25rem", background: "rgba(168,85,247,0.05)", border: "1px solid rgba(168,85,247,0.2)", borderRadius: "var(--radius-md)" }}>
            <h4 style={{ margin: "0 0 0.5rem", color: "#a855f7" }}>📤 Upload Health Data</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "0 0 0.75rem" }}>Upload a CSV, JSON, or Apple Health export to the active profile. Works with any device type.</p>

            {userDevice === "apple_watch" && (
              <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", background: "rgba(255,255,255,0.03)", padding: "0.75rem", borderRadius: "var(--radius-sm)", marginBottom: "0.75rem", lineHeight: 1.6 }}>
                <strong style={{ color: "var(--text-primary)" }}>How to export from Apple Health:</strong><br />
                1. Open <strong>Health</strong> app → tap your profile picture (top right)<br />
                2. Scroll down → <strong>Export All Health Data</strong> → Export<br />
                3. Share the <code>.zip</code> file to your laptop (AirDrop / iCloud Drive / email)<br />
                4. Upload the <code>.zip</code> below — no need to unzip
              </div>
            )}
            {userDevice === "oneplus" && (
              <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", background: "rgba(255,255,255,0.03)", padding: "0.75rem", borderRadius: "var(--radius-sm)", marginBottom: "0.75rem", lineHeight: 1.6 }}>
                <strong style={{ color: "var(--text-primary)" }}>How to export from OnePlus Health:</strong><br />
                OnePlus Health does not yet have a standard export format.<br />
                Please use the <strong>CSV template</strong> below to manually enter your data,<br />
                or export via Google Fit and upload the JSON.
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <a href="/api/wearable/upload-template" download style={{ fontSize: "0.8rem", color: "var(--accent-primary)", textDecoration: "none" }}>
                ⬇️ Download CSV template
              </a>
              <input type="file" accept=".zip,.xml,.csv,.json" id="health-upload-input" style={{ display: "none" }}
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  setUploadStatus({ uploading: true, message: `⏳ Uploading ${file.name}...` });
                  const form = new FormData();
                  form.append("file", file);
                  form.append("source", userDevice);
                  try {
                    const res = await fetch("/api/wearable/upload", { method: "POST", headers: authHeaders, body: form });
                    const result = await res.json();
                    if (result.errors?.length) {
                      setUploadStatus({ uploading: false, message: `✅ Saved ${result.rows_processed} days. ⚠️ ${result.errors.length} rows skipped.` });
                    } else {
                      setUploadStatus({ uploading: false, message: `✅ Imported ${result.rows_processed} days of health data.` });
                    }
                    fetchHistory();
                  } catch (err) {
                    setUploadStatus({ uploading: false, message: "❌ Upload failed. Check the file format." });
                  }
                  e.target.value = "";
                }}
              />
              <button onClick={() => document.getElementById("health-upload-input")?.click()} disabled={uploadStatus?.uploading}
                style={{ background: "var(--accent-primary)", color: "#fff", border: "none", padding: "0.75rem", borderRadius: "var(--radius-sm)", fontWeight: 600, cursor: uploadStatus?.uploading ? "not-allowed" : "pointer" }}>
                {uploadStatus?.uploading ? "Uploading..." : "Choose File & Upload"}
              </button>
              {uploadStatus && !uploadStatus.uploading && (
                <div style={{ padding: "0.75rem", borderRadius: "var(--radius-sm)", fontSize: "0.85rem",
                  background: uploadStatus.message.startsWith("✅") ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                  border: `1px solid ${uploadStatus.message.startsWith("✅") ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                  color: uploadStatus.message.startsWith("✅") ? "#22c55e" : "#ef4444"
                }}>{uploadStatus.message}</div>
              )}
            </div>
          </div>

          {/* ---- 4. Apple Health Live Push (iOS Shortcut) ---- */}
          {userDevice === "apple_watch" && (
            <AppleHealthPushSection userId={currentUserId} />
          )}

          {/* ---- 5. Simulation Mode ---- */}
          <div style={{ padding: "1.25rem", background: "rgba(99,102,241,0.05)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: "var(--radius-md)" }}>
            <h4 style={{ margin: "0 0 0.5rem", color: "var(--accent-primary)" }}>🧪 Debug & Simulation</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>Use simulated data to test the causal matrix with mock values.</p>
            <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
              <label className="switch">
                <input type="checkbox" checked={simulateMode} onChange={e => setSimulatedMode(e.target.checked)} />
                <span className="slider round"></span>
              </label>
              <span style={{ fontSize: "0.9rem", color: simulateMode ? "var(--accent-primary)" : "var(--text-secondary)", fontWeight: 600 }}>
                {simulateMode ? "Simulated Mode ACTIVE" : "Simulated Mode DISABLED"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ============ ADMIN TAB ============ */}
      {activeTab === 'admin' && <AdminPanel />}
    </div>
  );
}

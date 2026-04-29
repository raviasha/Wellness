"use client";

import React, { useState, useEffect } from 'react';

export default function AdminPanel() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeSection, setActiveSection] = useState<'users' | 'matrix' | 'calibration' | 'rawdata'>('users');
  
  // Calibration state
  const [selectedUserId, setSelectedUserId] = useState<number>(1);
  const [calibResult, setCalibResult] = useState<any>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState<any>(null);
  const [calibError, setCalibError] = useState("");
  
  // New Transparency & Approval state
  const [config, setConfig] = useState<any>(null);
  const [draftPersona, setDraftPersona] = useState<any>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [approvalLoading, setApprovalLoading] = useState(false);
  const [approvalStatus, setApprovalStatus] = useState<any>(null);
  
  // Raw data inspector state
  const [rawSyncData, setRawSyncData] = useState<any[]>([]);
  const [rawSyncLoading, setRawSyncLoading] = useState(false);
  const [rawSyncUserId, setRawSyncUserId] = useState<number>(1);

  // Activity backfill state
  const [backfillActLoading, setBackfillActLoading] = useState(false);
  const [backfillActResult, setBackfillActResult] = useState<any>(null);

  const runBackfillActivities = async () => {
    setBackfillActLoading(true);
    setBackfillActResult(null);
    try {
      const res = await fetch("/api/admin/backfill-activities", { method: "POST" });
      const json = await res.json();
      setBackfillActResult(json);
      await fetchAdminData(); // refresh matrix
    } catch (e: any) {
      setBackfillActResult({ error: e.message });
    } finally {
      setBackfillActLoading(false);
    }
  };

  useEffect(() => {
    fetchAdminData();
    fetchAdminConfig();
  }, []);

  const fetchAdminConfig = async () => {
    try {
      const res = await fetch("/api/admin/config");
      if (res.ok) setConfig(await res.json());
    } catch (e) { console.error(e); }
  };

  const fetchAdminData = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/admin/data");
      if (res.ok) {
        const d = await res.json();
        setData(d);
        if (d.users.length > 0) setSelectedUserId(d.users[0].id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCalibrate = async () => {
    setCalibrating(true);
    setCalibError("");
    setCalibResult(null);
    try {
      const res = await fetch("/api/calibrate", {
        method: "POST",
        headers: { 'X-User-ID': selectedUserId.toString() }
      });
      const result = await res.json();
      if (!res.ok) {
        setCalibError(result.detail || "Calibration failed");
      } else {
        setCalibResult(result);
        setDraftPersona(result.params); // Init edit form with result
        setIsEditing(true);
      }
    } catch (err: any) {
      setCalibError(err.message);
    } finally {
      setCalibrating(false);
    }
  };

  const handleApprove = async () => {
    setApprovalLoading(true);
    setApprovalStatus(null);
    try {
      const res = await fetch("/api/persona/approve", {
        method: "POST",
        headers: { 
          'Content-Type': 'application/json',
          'X-User-ID': selectedUserId.toString() 
        },
        body: JSON.stringify(draftPersona)
      });
      const result = await res.json();
      if (res.ok) {
        setApprovalStatus({ status: "success", message: result.message });
        setIsEditing(false);
        fetchAdminData(); // Refresh user list to show approved toggle
      } else {
        setApprovalStatus({ status: "error", message: result.detail });
      }
    } catch (err: any) {
      setApprovalStatus({ status: "error", message: err.message });
    } finally {
      setApprovalLoading(false);
    }
  };

  const handleTrainNN = async () => {
    setTraining(true);
    setTrainStatus(null);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { 'X-User-ID': selectedUserId.toString() }
      });
      if (!res.ok) {
        const err = await res.json();
        setTrainStatus({ status: "error", message: err.detail });
        setTraining(false);
        return;
      }

      // Poll for completion
      const poll = async (): Promise<void> => {
        const statusRes = await fetch("/api/train/status", {
          headers: { 'X-User-ID': selectedUserId.toString() }
        });
        const status = await statusRes.json();
        setTrainStatus(status);

        if (status.status === "running") {
          await new Promise(r => setTimeout(r, 3000));
          return poll();
        } else {
          setTraining(false);
        }
      };
      await poll();
    } catch (err: any) {
      setTrainStatus({ status: "error", message: err.message });
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-secondary)" }}>
        <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>⏳</div>
        Loading admin data...
      </div>
    );
  }

  if (!data) {
    return <div style={{ padding: "2rem", color: "var(--danger)" }}>Failed to load admin data.</div>;
  }

  const tabStyle = (tab: string) => ({
    padding: "0.5rem 1.25rem",
    background: activeSection === tab ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
    border: activeSection === tab ? '1px solid rgba(99, 102, 241, 0.3)' : '1px solid transparent',
    color: activeSection === tab ? 'var(--accent-primary)' : 'var(--text-secondary)',
    cursor: "pointer" as const,
    fontWeight: 600,
    borderRadius: "var(--radius-sm)",
    fontSize: "0.85rem",
    transition: "all 0.2s ease",
  });

  const thStyle: React.CSSProperties = {
    textAlign: "left",
    padding: "0.75rem 1rem",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    color: "var(--text-secondary)",
    fontSize: "0.75rem",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    fontWeight: 600,
  };

  const tdStyle: React.CSSProperties = {
    padding: "0.6rem 1rem",
    borderBottom: "1px solid rgba(255,255,255,0.04)",
    fontSize: "0.85rem",
    color: "var(--text-primary)",
  };

  const userMap: Record<number, string> = {};
  data.users.forEach((u: any) => { userMap[u.id] = u.username; });

  const r2Color = (val: number) => val > 0.7 ? "#22c55e" : val > 0.4 ? "#f59e0b" : "#ef4444";

  return (
    <div>
      {/* Summary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "2rem" }}>
        {[
          { label: "Users", count: data.counts.users, icon: "👤", color: "#6366f1" },
          { label: "Profiles", count: data.counts.profiles, icon: "📋", color: "#a855f7" },
          { label: "Garmin Syncs", count: data.counts.syncs, icon: "⌚", color: "#22c55e" },
          { label: "Manual Logs", count: data.counts.logs, icon: "📝", color: "#f59e0b" },
        ].map((card) => (
          <div key={card.label} style={{
            background: `linear-gradient(135deg, ${card.color}15 0%, ${card.color}05 100%)`,
            border: `1px solid ${card.color}30`,
            borderRadius: "var(--radius-md)",
            padding: "1.25rem",
            textAlign: "center",
          }}>
            <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>{card.icon}</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, color: card.color }}>{card.count}</div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>{card.label}</div>
          </div>
        ))}
      </div>

      {/* Section Tabs */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem", flexWrap: "wrap" }}>
        <button onClick={() => setActiveSection('users')} style={tabStyle('users')}>👤 Users</button>
        <button onClick={() => setActiveSection('matrix')} style={tabStyle('matrix')}>📉 Daily Causal Matrix</button>
        <button onClick={() => setActiveSection('calibration')} style={tabStyle('calibration')}>🧬 Calibration & Training</button>
        <button onClick={() => setActiveSection('rawdata')} style={tabStyle('rawdata')}>🔬 Raw Sync Data</button>
        <button onClick={runBackfillActivities} disabled={backfillActLoading} style={{ ...tabStyle('refresh'), color: backfillActResult?.errors?.length ? "#f97316" : "var(--accent-primary)" }}>
          {backfillActLoading ? "⏳ Fetching..." : "🏃 Backfill Activities"}
        </button>
        {backfillActResult && !backfillActLoading && (
          <span style={{ fontSize: "0.75rem", color: backfillActResult.error ? "#ef4444" : "var(--text-secondary)", alignSelf: "center" }}>
            {backfillActResult.error ? `Error: ${backfillActResult.error}` : `✓ ${backfillActResult.updated_rows} updated, ${backfillActResult.vo2_ffilled} VO2 ffilled${backfillActResult.errors?.length ? ` (${backfillActResult.errors.length} err)` : ""}`}
          </span>
        )}
        <button onClick={fetchAdminData} style={{ ...tabStyle('refresh'), marginLeft: "auto", color: "var(--text-secondary)" }}>🔄 Refresh</button>
      </div>

      {/* === CALIBRATION & TRAINING SECTION === */}
      {activeSection === 'calibration' && (
        <div>
          {/* User Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Select User:</label>
            <select 
              value={selectedUserId} 
              onChange={(e) => { setSelectedUserId(parseInt(e.target.value)); setCalibResult(null); setTrainStatus(null); }}
              style={{ background: "var(--bg-card)", color: "#fff", border: "1px solid var(--border-color)", padding: "0.5rem 1rem", borderRadius: "var(--radius-sm)", outline: "none" }}
            >
              {data.users.map((u: any) => <option key={u.id} value={u.id}>{u.username} (ID: {u.id})</option>)}
            </select>
            <button 
              onClick={handleCalibrate} 
              disabled={calibrating}
              style={{ 
                background: "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)", color: "#fff", border: "none", 
                padding: "0.6rem 1.5rem", borderRadius: "var(--radius-sm)", fontWeight: 700, cursor: "pointer",
                opacity: calibrating ? 0.6 : 1
              }}
            >
              {calibrating ? "⏳ Running Regression..." : "🧬 Deep Calibration"}
            </button>
          </div>

          {calibError && (
            <div style={{ padding: "0.75rem", background: "rgba(239,68,68,0.1)", color: "#ef4444", borderRadius: "var(--radius-sm)", marginBottom: "1rem" }}>
              ❌ {calibError}
            </div>
          )}

             {calibResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
              
              {/* === TRANSPARENCY LAYER: ARCHITECTURE === */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                <div className="premium-card" style={{ padding: "1.5rem", background: "rgba(255,255,255,0.02)" }}>
                  <h4 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "var(--accent-primary)" }}>🔍 Model Architecture (Lagged Regression)</h4>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                    We use a 1-day lag model: <strong>Actions on Day T</strong> are mapped to <strong>Outcome Changes on Day T+1</strong>.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "0.5rem" }}>Input Features (Day T)</div>
                      <ul style={{ fontSize: "0.8rem", color: "var(--text-primary)", paddingLeft: "1.25rem" }}>
                        <li>Sleep Duration (h)</li>
                        <li>Steps</li>
                        <li>Active Minutes</li>
                        <li>Active Calories</li>
                      </ul>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "0.5rem" }}>Outcome Variables (Δ T+1)</div>
                      <ul style={{ fontSize: "0.8rem", color: "var(--text-primary)", paddingLeft: "1.25rem" }}>
                        <li>Δ Sleep Score</li>
                        <li>Δ Heart Rate Variability (HRV)</li>
                        <li>Δ Resting Heart Rate (RHR)</li>
                        <li>Δ Body Battery</li>
                        <li>Δ Garmin Stress (Physiological)</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="premium-card" style={{ padding: "1.5rem", background: "rgba(255,255,255,0.02)" }}>
                  <h4 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "var(--accent-secondary)" }}>⚖️ AI Incentive Structure (Reward Function)</h4>
                  {config && (
                    <>
                      <div style={{ fontSize: "0.8rem", marginBottom: "1rem", display: "flex", gap: "1.5rem" }}>
                        <div style={{ padding: "0.5rem", background: "rgba(99,102,241,0.1)", borderRadius: "4px" }}>
                          <div style={{ fontSize: "0.65rem", color: "var(--text-secondary)" }}>CHANGE WEIGHT</div>
                          <div style={{ fontWeight: 700 }}>{config.delta_weight_ratio * 100}%</div>
                        </div>
                        <div style={{ padding: "0.5rem", background: "rgba(168,85,247,0.1)", borderRadius: "4px" }}>
                          <div style={{ fontSize: "0.65rem", color: "var(--text-secondary)" }}>STATE QUALITY</div>
                          <div style={{ fontWeight: 700 }}>{config.state_weight_ratio * 100}%</div>
                        </div>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "0.5rem" }}>
                        {Object.entries(config.goal_weights[data.profiles.find((p:any)=>p.user_id===selectedUserId)?.goal || 'stress_management']).slice(0, 6).map(([k, v]: any) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", padding: "0.25rem 0", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                            <span style={{ color: "var(--text-secondary)" }}>{k.replace('_', ' ')}</span>
                            <span style={{ fontWeight: 700, color: "var(--accent-secondary)" }}>{(v * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* === REGRESSION DETAILS === */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "1.5rem" }}>
                <div style={{ 
                  background: "linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.05) 100%)",
                  border: "1px solid rgba(99,102,241,0.2)", borderRadius: "var(--radius-md)", padding: "1.5rem",
                  textAlign: "center"
                }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "0.5rem" }}>Digital Twin Alignment (R²)</div>
                  <div style={{ fontSize: "3rem", fontWeight: 800, color: r2Color(calibResult.r2_overall) }}>
                    {(calibResult.r2_overall * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
                    {calibResult.samples} samples • {calibResult.method}
                  </div>
                  
                  <div style={{ marginTop: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {Object.entries(calibResult.r2_per_outcome).map(([name, val]: any) => (
                      <div key={name} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                        <span style={{ color: "var(--text-secondary)" }}>{name}</span>
                        <span style={{ fontWeight: 700, color: r2Color(val) }}>{(val * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 style={{ margin: "0 0 0.75rem", color: "var(--text-secondary)", fontSize: "0.85rem", textTransform: "uppercase" }}>
                    Regression Weight Matrix (Learned Physiology)
                  </h4>
                  <div style={{ overflowX: "auto", background: "rgba(0,0,0,0.2)", borderRadius: "var(--radius-sm)" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Feature</th>
                          {calibResult.outcomes.map((o: string) => (
                            <th key={o} style={thStyle}>{o}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {calibResult.weights.map((row: any, i: number) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontWeight: 600 }}>{row.feature}</td>
                            {calibResult.outcomes.map((o: string) => {
                              const v = row[o];
                              return (
                                <td key={o} style={{ ...tdStyle, color: v > 0 ? "#22c55e" : v < 0 ? "#ef4444" : "var(--text-secondary)", fontFamily: "monospace" }}>
                                  {v > 0 ? "+" : ""}{v}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* === MAPPING & EDITING === */}
              <div className="premium-card" style={{ padding: "1.5rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                  <h4 style={{ margin: 0, fontSize: "1rem", color: "var(--accent-primary)" }}>🛠️ Simulator Rule Configuration (Draft)</h4>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                    {isEditing ? "✨ Manual Tweak Mode Active" : "🔒 View Mode"}
                  </div>
                </div>
                
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
                  Review how the regression weights were mapped to Simulator rules. 
                  You can manually adjust these values if the regression produced outliers.
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "1rem 2rem" }}>
                  {draftPersona && Object.entries(draftPersona).map(([key, value]: [string, any]) => (
                    <div key={key} style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                        <label style={{ fontSize: "0.8rem", color: "var(--text-primary)", fontWeight: 500 }}>{key.replace(/_/g, ' ').toUpperCase()}</label>
                        <span style={{ fontSize: "0.65rem", color: "var(--accent-primary)", fontStyle: "italic" }}>
                          {key.includes('hrv') ? 'Impacts HRV' : key.includes('rhr') ? 'Impacts RHR' : 'Impacts Biomarker'}
                        </span>
                      </div>
                      <input 
                        type="number"
                        step="0.01"
                        value={value}
                        onChange={(e) => setDraftPersona({...draftPersona, [key]: parseFloat(e.target.value)})}
                        readOnly={!isEditing}
                        className="premium-input"
                        style={{ padding: "0.5rem", fontSize: "0.9rem", width: "100%", background: isEditing ? 'rgba(99,102,241,0.05)' : 'transparent' }}
                      />
                    </div>
                  ))}
                </div>

                {/* APPROVAL ACTIONS */}
                <div style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", gap: "1rem" }}>
                  <button 
                    onClick={handleApprove}
                    disabled={approvalLoading}
                    style={{ 
                      background: "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)", color: "#fff", border: "none",
                      padding: "0.75rem 2rem", borderRadius: "var(--radius-sm)", fontWeight: 700, cursor: "pointer",
                      boxShadow: "0 4px 12px rgba(99,102,241,0.3)"
                    }}
                  >
                    {approvalLoading ? "⏳ Saving..." : "✅ Approve & Finalize Simulator"}
                  </button>
                  <button 
                    onClick={() => setCalibResult(null)}
                    style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.1)", color: "var(--text-secondary)", padding: "0.75rem 1.5rem", borderRadius: "var(--radius-sm)", cursor: "pointer" }}
                  >
                    Cancel
                  </button>
                  
                  {approvalStatus && (
                    <span style={{ color: approvalStatus.status === 'success' ? '#22c55e' : '#ef4444', fontSize: '0.85rem' }}>
                      {approvalStatus.status === 'success' ? '✔ Approved!' : `❌ ${approvalStatus.message}`}
                    </span>
                  )}
                </div>
              </div>

              {/* === TRAINING (Guarded) === */}
              <div style={{ 
                background: "linear-gradient(135deg, rgba(34,197,94,0.05) 0%, rgba(34,197,94,0.02) 100%)",
                border: "1px solid rgba(34,197,94,0.2)", borderRadius: "var(--radius-md)", padding: "1.5rem",
                display: "flex", alignItems: "center", justifyContent: "space-between",
                opacity: approvalStatus?.status === 'success' ? 1 : 0.6
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                    <h4 style={{ margin: 0, fontSize: "1.1rem" }}>🧠 Neural Network Training</h4>
                    {approvalStatus?.status === 'success' ? (
                      <span style={{ fontSize: "0.65rem", background: "#22c55e", color: "#fff", padding: "0.1rem 0.4rem", borderRadius: "4px", fontWeight: 700 }}>SIMULATOR APPROVED</span>
                    ) : (
                      <span style={{ fontSize: "0.65rem", background: "#ef4444", color: "#fff", padding: "0.1rem 0.4rem", borderRadius: "4px", fontWeight: 700 }}>AWAITING APPROVAL</span>
                    )}
                  </div>
                  <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem", color: "var(--text-secondary)", maxWidth: "500px" }}>
                    Train the RL agent using your validated Digital Twin rules. Training takes ~30 seconds for 10,000 steps.
                  </p>
                </div>
                <button 
                  onClick={handleTrainNN} 
                  disabled={training || approvalStatus?.status !== 'success'}
                  style={{ 
                    background: training || approvalStatus?.status !== 'success' ? "var(--text-secondary)" : "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)", 
                    color: "#fff", border: "none", 
                    padding: "0.85rem 2.5rem", borderRadius: "var(--radius-sm)", fontWeight: 700, cursor: "pointer",
                    opacity: training || approvalStatus?.status !== 'success' ? 0.5 : 1, whiteSpace: "nowrap"
                  }}
                >
                  {training ? "⏳ Training agent..." : "🚀 Launch Training Cycle"}
                </button>
              </div>

              {trainStatus && (
                <div style={{ 
                  padding: "1rem", borderRadius: "var(--radius-sm)", fontSize: "0.9rem",
                  background: trainStatus.status === "complete" ? "rgba(34,197,94,0.1)" : trainStatus.status === "error" ? "rgba(239,68,68,0.1)" : "rgba(99,102,241,0.1)",
                  color: trainStatus.status === "complete" ? "#22c55e" : trainStatus.status === "error" ? "#ef4444" : "var(--accent-primary)",
                  border: `1px solid ${trainStatus.status === "complete" ? "rgba(34,197,94,0.3)" : trainStatus.status === "error" ? "rgba(239,68,68,0.3)" : "rgba(99,102,241,0.3)"}`,
                }}>
                  {trainStatus.status === "complete" ? "✅ Success!" : trainStatus.status === "error" ? "❌ Error" : "⏳ Running:"} {trainStatus.message}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Users Table */}
      {activeSection === 'users' && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={thStyle}>ID</th>
                <th style={thStyle}>Username</th>
                <th style={thStyle}>Garmin Email</th>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Age</th>
                <th style={thStyle}>Goal</th>
                <th style={thStyle}>Compliance</th>
                <th style={thStyle}>Digital Twin</th>
                <th style={thStyle}>CreatedAt</th>
              </tr>
            </thead>
            <tbody>
              {data.users.map((u: any) => {
                const profile = data.profiles.find((p: any) => p.user_id === u.id);
                return (
                  <tr key={u.id}>
                    <td style={tdStyle}><span style={{ background: "rgba(99,102,241,0.15)", padding: "0.15rem 0.5rem", borderRadius: "4px", fontSize: "0.8rem" }}>{u.id}</span></td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{u.username}</td>
                    <td style={tdStyle}>{u.garmin_email || <span style={{ color: "var(--text-secondary)", fontStyle: "italic" }}>Not set</span>}</td>
                    <td style={tdStyle}>{profile?.name || "—"}</td>
                    <td style={tdStyle}>{profile?.age || "—"}</td>
                    <td style={tdStyle}>
                      {profile?.goal ? (
                        <span style={{ background: "rgba(34,197,94,0.1)", color: "var(--success)", padding: "0.15rem 0.5rem", borderRadius: "4px", fontSize: "0.8rem" }}>{profile.goal}</span>
                      ) : "—"}
                    </td>
                    <td style={tdStyle}>{profile?.compliance_rate ? `${(profile.compliance_rate * 100).toFixed(0)}%` : "—"}</td>
                    <td style={tdStyle}>
                      {profile?.simulator_approved ? (
                        <span style={{ color: "#22c55e", fontWeight: 700, fontSize: "0.75rem" }}>✅ APPROVED</span>
                      ) : (
                        <span style={{ color: "#ef4444", fontWeight: 700, fontSize: "0.75rem" }}>⚠️ PENDING</span>
                      )}
                    </td>
                    <td style={{ ...tdStyle, color: "var(--text-secondary)", fontSize: "0.8rem" }}>{u.created_at?.split('.')[0]}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Daily Causal Matrix Table */}
      {activeSection === 'matrix' && (
        <div style={{ overflowX: "auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
             <h3 style={{ margin: 0, fontSize: "1.1rem" }}>📅 Longitudinal Behavior & Bio-Response Ledger</h3>
             <div>
                <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginRight: "0.5rem" }}>Filter by User:</label>
                <select 
                  value={selectedUserId} 
                  onChange={(e) => setSelectedUserId(parseInt(e.target.value))}
                  style={{ background: "var(--bg-card)", color: "#fff", border: "1px solid var(--border-color)", padding: "0.4rem", borderRadius: "var(--radius-sm)" }}
                >
                  {data.users.map((u: any) => <option key={u.id} value={u.id}>{u.username}</option>)}
                </select>
             </div>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "rgba(255,255,255,0.02)" }}>
                <th style={thStyle}>Date</th>
                {/* ── Behavioral Inputs (X) ── */}
                <th style={{ ...thStyle, color: "var(--accent-primary)", borderLeft: "2px solid rgba(99,102,241,0.3)" }}>Sleep (h)</th>
                <th style={{ ...thStyle, color: "var(--accent-primary)" }}>Bedtime (IST)</th>
                <th style={{ ...thStyle, color: "var(--accent-primary)" }}>Ex. Type</th>
                <th style={{ ...thStyle, color: "var(--accent-primary)" }}>Act Cal</th>
                <th style={{ ...thStyle, color: "var(--accent-primary)" }}>Intensity</th>
                {/* ── Biological Outcomes (Y) ── */}
                <th style={{ ...thStyle, color: "var(--success)", borderLeft: "2px solid rgba(34,197,94,0.3)" }}>Sleep Score</th>
                <th style={{ ...thStyle, color: "var(--success)" }}>HRV</th>
                <th style={{ ...thStyle, color: "var(--success)" }}>RHR</th>
                <th style={{ ...thStyle, color: "var(--success)" }}>Stress</th>
                <th style={{ ...thStyle, color: "var(--success)" }}>Body Battery</th>
                <th style={{ ...thStyle, color: "#14b8a6" }}>Stage Qual %</th>
                <th style={{ ...thStyle, color: "#f97316" }}>VO2 Max</th>
              </tr>
            </thead>
            <tbody>
              {(() => {
                const userSyncs = data.syncs.filter((s: any) => s.user_id === selectedUserId);
                const userLogs = data.logs.filter((l: any) => l.user_id === selectedUserId);
                
                const rawSyncs: any = {};
                const rawLogs: any = {};
                const datesSet = new Set<string>();

                userSyncs.forEach((s: any) => {
                  rawSyncs[s.sync_date] = s;
                  datesSet.add(s.sync_date);
                });
                userLogs.forEach((l: any) => {
                  if (!rawLogs[l.log_date]) rawLogs[l.log_date] = [];
                  rawLogs[l.log_date].push(l);
                  datesSet.add(l.log_date);
                });

                const sortedDates = Array.from(datesSet).sort((a,b) => b.localeCompare(a));

                // Pre-compute forward-filled vo2_max (ascending order)
                const sortedAsc = [...sortedDates].sort();
                const vo2Ffill: Record<string, number> = {};
                let lastVo2: number | null = null;
                sortedAsc.forEach(d => {
                  const s = rawSyncs[d] || {};
                  if (s.vo2_max != null) lastVo2 = s.vo2_max;
                  if (lastVo2 != null) vo2Ffill[d] = lastVo2;
                });
                
                return sortedDates.map(date => {
                  const dObj = new Date(date);
                  const prevDate = new Date(dObj.getTime() - 86400000).toISOString().split('T')[0];
                  
                  const todaySync = rawSyncs[date] || {};
                  const todayLogs = rawLogs[date] || [];
                  const prevSync = rawSyncs[prevDate] || {};
                  const prevLogs = rawLogs[prevDate] || [];

                  const row: any = { date };

                  // Helper to parse raw_payload safely
                  const parseRaw = (sync: any) => {
                    try { return typeof sync.raw_payload === 'string' ? JSON.parse(sync.raw_payload) : (sync.raw_payload || {}); }
                    catch { return {}; }
                  };

                  const aggregateActivities = (acts: any[]) => {
                    if (!Array.isArray(acts) || acts.length === 0) return null;

                    let totalSeconds = 0;
                    const typeMinutes: Record<string, number> = {};

                    acts.forEach((a: any) => {
                      const sec = Number(a?.duration || 0);
                      if (sec > 0) totalSeconds += sec;

                      const typeKey = String(a?.activityType?.typeKey || 'other');
                      const mins = sec > 0 ? sec / 60 : 0;
                      typeMinutes[typeKey] = (typeMinutes[typeKey] || 0) + mins;
                    });

                    const sortedTypes = Object.entries(typeMinutes)
                      .sort((a, b) => b[1] - a[1])
                      .map(([k]) => k.replace(/_/g, ' '));

                    const typeLabel =
                      sortedTypes.length <= 2
                        ? sortedTypes.join(' + ')
                        : `${sortedTypes.slice(0, 2).join(' + ')} +${sortedTypes.length - 2}`;

                    return {
                      typeLabel: typeLabel || 'none',
                      totalMinutes: Math.round(totalSeconds / 60),
                    };
                  };

                  // === OUTCOMES (Y) from Today ===
                  row.hrv = todaySync.hrv_rmssd;
                  row.rhr = todaySync.resting_hr;
                  row.sleep_score = todaySync.sleep_score;
                  row.stress_avg = todaySync.stress_avg;
                  row.body_battery = todaySync.recovery_score; // DB column is recovery_score
                  row.sleep_stage_quality = todaySync.sleep_stage_quality;
                  row.vo2_max = todaySync.vo2_max ?? vo2Ffill[date] ?? null;

                  // === INPUTS (X) from Yesterday (Lagged) ===
                  row.active_calories = prevSync.active_calories;
                  row.active_minutes = prevSync.active_minutes;
                  row.steps = prevSync.steps;
                  row.bedtime_hour = prevSync.sleep_start_hour;
                  row.exercise_type = prevSync.exercise_type;
                  row.exercise_duration = prevSync.exercise_duration_minutes;

                  // Aggregate exercise feature from all sessions for Day T.
                  const rawPrevForActs = parseRaw(prevSync);
                  const aggActs = aggregateActivities(rawPrevForActs?.activities || []);
                  if (aggActs) {
                    row.exercise_type = aggActs.typeLabel;
                    row.exercise_duration = aggActs.totalMinutes;
                  }

                  // ── Sleep duration: stored column → raw payload fallback ──
                  if (todaySync.sleep_duration_hours) {
                    row.sleep_h = todaySync.sleep_duration_hours;
                  } else {
                    const rawToday = parseRaw(todaySync);
                    const sleepObj = rawToday?.sleep || {};
                    const dto = sleepObj?.dailySleepDTO || {};
                    const dur = sleepObj?.durationInSeconds || dto?.sleepDurationInSeconds || dto?.sleepTimeSeconds;
                    if (dur) row.sleep_h = (dur / 3600).toFixed(1);
                  }

                  // ── Bedtime: stored column → raw payload fallback ──
                  // sleepStartTimestampLocal = GMT_ms + IST_offset_ms
                  // Using getUTCHours() on it gives IST wall-clock hours directly
                  if (row.bedtime_hour == null) {
                    const rawPrev = parseRaw(prevSync);
                    const dto = rawPrev?.sleep?.dailySleepDTO || {};
                    const ts = dto?.sleepStartTimestampLocal;
                    if (ts) {
                      const d = new Date(parseInt(ts));
                      row.bedtime_hour = d.getUTCHours() + d.getUTCMinutes() / 60;
                    }
                  }

                  // ── body_battery: raw payload fallback ──
                  if (row.body_battery == null) {
                    const rawToday = parseRaw(todaySync);
                    const bb = rawToday?.body_battery;
                    if (Array.isArray(bb) && bb.length > 0) {
                      row.body_battery = bb[bb.length - 1]?.charged ?? bb[0]?.charged;
                    } else if (bb && typeof bb === 'object') {
                      row.body_battery = bb.charged ?? bb.latestValue;
                    }
                  }

                  // ── sleep_stage_quality: raw payload fallback ──
                  if (row.sleep_stage_quality == null) {
                    const rawToday = parseRaw(todaySync);
                    const dto = rawToday?.sleep?.dailySleepDTO || {};
                    const deep = dto?.deepSleepSeconds || 0;
                    const rem = dto?.remSleepSeconds || 0;
                    const light = dto?.lightSleepSeconds || 0;
                    const awake = dto?.awakeSleepSeconds || 0;
                    const total = deep + rem + light + awake;
                    if (total > 0) row.sleep_stage_quality = ((deep + rem) / total * 100).toFixed(1);
                  }

                  // ── vo2_max: raw payload fallback ──
                  if (row.vo2_max == null) {
                    const rawToday = parseRaw(todaySync);
                    const mm = rawToday?.max_metrics;
                    if (Array.isArray(mm) && mm.length > 0) {
                      row.vo2_max = mm[0]?.vo2Max ?? mm[0]?.vo2max ?? mm[0]?.generic?.vo2MaxPreciseValue;
                    } else if (mm && typeof mm === 'object') {
                      row.vo2_max = mm.vo2Max ?? mm.vo2max ?? mm?.generic?.vo2MaxPreciseValue;
                    }
                  }

                  // ── exercise fields fallback (only when no aggregated data available) ──
                  if (!row.exercise_type || row.exercise_duration == null) {
                    const rawPrev = parseRaw(prevSync);
                    const acts: any[] = rawPrev?.activities || [];
                    if (acts.length > 0) {
                      const agg = aggregateActivities(acts);
                      if (agg) {
                        if (!row.exercise_type) row.exercise_type = agg.typeLabel;
                        if (row.exercise_duration == null) row.exercise_duration = agg.totalMinutes;
                      }
                    }
                  }

                  // ── steps: raw payload fallback ──
                  if (row.steps == null) {
                    const rawPrev = parseRaw(prevSync);
                    row.steps = rawPrev?.steps?.totalSteps ?? null;
                  }

                  prevLogs.forEach((l: any) => {
                    if (l.log_type === 'proxy_sleep') row.sleep_h = l.value;
                  });

                  return (
                    <tr key={date}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: "nowrap" }}>
                        <div style={{ color: "var(--success)" }}>{date}</div>
                        <div style={{ fontSize: "0.65rem", color: "var(--accent-primary)", marginTop: "2px" }}>← {prevDate}</div>
                      </td>
                      {/* ── Inputs (X) ── */}
                      <td style={{ ...tdStyle, borderLeft: "2px solid rgba(99,102,241,0.1)" }}>{row.sleep_h || "—"}</td>
                      <td style={tdStyle}>{row.bedtime_hour != null ? `${Math.floor(row.bedtime_hour)}:${String(Math.round((row.bedtime_hour % 1) * 60)).padStart(2,'0')}` : "—"}</td>
                      <td style={{ ...tdStyle, textTransform: "capitalize" }}>{(row.exercise_type && row.exercise_type !== 'none') ? row.exercise_type.replace(/_/g, ' ') : "—"}</td>
                      <td style={tdStyle}>{row.active_calories || "—"}</td>
                      <td style={tdStyle}>{row.active_minutes ? `${row.active_minutes}m` : "—"}</td>
                      {/* ── Outcomes (Y) ── */}
                      <td style={{ ...tdStyle, borderLeft: "2px solid rgba(34,197,94,0.1)" }}>{row.sleep_score || "—"}</td>
                      <td style={{ ...tdStyle, color: "var(--success)" }}>{row.hrv || "—"}</td>
                      <td style={tdStyle}>{row.rhr || "—"}</td>
                      <td style={tdStyle}>{row.stress_avg || "—"}</td>
                      <td style={tdStyle}>{row.body_battery || "—"}</td>
                      <td style={{ ...tdStyle, color: "#14b8a6" }}>{row.sleep_stage_quality != null ? `${row.sleep_stage_quality}%` : "—"}</td>
                      <td style={{ ...tdStyle, color: "#f97316" }}>{row.vo2_max != null ? (typeof row.vo2_max === "number" ? row.vo2_max.toFixed(1) : row.vo2_max) : "—"}</td>
                    </tr>
                  );
                });
              })()}
            </tbody>
          </table>
          {data.syncs.length === 0 && data.logs.length === 0 && (
            <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-secondary)" }}>No causal data found for this user.</div>
          )}
        </div>
      )}

      {/* === RAW SYNC DATA INSPECTOR === */}
      {activeSection === 'rawdata' && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>User:</label>
            <select
              value={rawSyncUserId}
              onChange={(e) => setRawSyncUserId(parseInt(e.target.value))}
              style={{ background: "var(--bg-card)", color: "#fff", border: "1px solid var(--border-color)", padding: "0.5rem 1rem", borderRadius: "var(--radius-sm)", outline: "none" }}
            >
              {(data?.users || []).map((u: any) => (
                <option key={u.id} value={u.id}>{u.username} (ID {u.id})</option>
              ))}
            </select>
            <button
              onClick={async () => {
                setRawSyncLoading(true);
                try {
                  const res = await fetch("/api/debug/raw-sync", { headers: { "X-User-ID": rawSyncUserId.toString() } });
                  if (res.ok) setRawSyncData(await res.json());
                } catch (e) { console.error(e); }
                setRawSyncLoading(false);
              }}
              style={{ background: "var(--accent-primary)", color: "#fff", border: "none", padding: "0.5rem 1rem", borderRadius: "var(--radius-sm)", cursor: "pointer", fontWeight: 600 }}
            >
              {rawSyncLoading ? "Loading..." : "🔍 Inspect"}
            </button>
          </div>

          {rawSyncData.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                Compares what Garmin returned (raw) vs what was extracted into DB columns. Mismatches are highlighted.
              </p>
              {rawSyncData.map((row: any, i: number) => {
                const ext = row.extracted || {};
                const rawH = row.raw_highlights || {};
                const fields = [
                  { label: "HRV (RMSSD)", extracted: ext.hrv_rmssd, raw: rawH.hrv_lastNightAvg },
                  { label: "Resting HR", extracted: ext.resting_hr, raw: rawH.resting_hr_raw },
                  { label: "Recovery", extracted: ext.recovery_score, raw: rawH.body_battery_raw },
                  { label: "Sleep Score", extracted: ext.sleep_score, raw: rawH.sleep_score_raw },
                  { label: "Sleep Hours", extracted: ext.sleep_duration_hours, raw: rawH.sleep_duration_hours_raw },
                  { label: "Stress Avg", extracted: ext.stress_avg, raw: rawH.stress_avgStressLevel },
                  { label: "Active Mins", extracted: ext.active_minutes, raw: rawH.intensity_minutes_raw },
                  { label: "Active Cals", extracted: ext.active_calories, raw: rawH.active_calories_raw },
                  { label: "Steps", extracted: ext.steps, raw: rawH.steps_raw },
                  { label: "SpO2", extracted: ext.spo2, raw: rawH.spo2_raw },
                  { label: "Respiration", extracted: ext.respiration_rate, raw: rawH.respiration_raw },
                ];
                return (
                  <div key={i} style={{ marginBottom: "1.5rem", background: "rgba(255,255,255,0.02)", borderRadius: "var(--radius-sm)", padding: "1rem", border: "1px solid var(--border-color)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                      <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>📅 {row.sync_date}</span>
                      <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Source: {row.source} · Synced: {row.created_at}</span>
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left", padding: "0.4rem", color: "var(--text-secondary)", borderBottom: "1px solid var(--border-color)" }}>Metric</th>
                          <th style={{ textAlign: "right", padding: "0.4rem", color: "#6366f1", borderBottom: "1px solid var(--border-color)" }}>DB Extracted</th>
                          <th style={{ textAlign: "right", padding: "0.4rem", color: "#f59e0b", borderBottom: "1px solid var(--border-color)" }}>Raw Payload</th>
                          <th style={{ textAlign: "center", padding: "0.4rem", borderBottom: "1px solid var(--border-color)" }}>Match?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fields.map((f, fi) => {
                          const extVal = f.extracted != null ? String(f.extracted) : "—";
                          const rawVal = f.raw != null ? String(f.raw) : "—";
                          const bothNull = extVal === "—" && rawVal === "—";
                          const numMatch = f.extracted != null && f.raw != null && Number(f.extracted) === Number(f.raw);
                          const match = f.raw == null || extVal === rawVal || bothNull || numMatch;
                          return (
                            <tr key={fi} style={{ background: !match ? "rgba(239, 68, 68, 0.08)" : "transparent" }}>
                              <td style={{ padding: "0.35rem 0.4rem", color: "var(--text-primary)" }}>{f.label}</td>
                              <td style={{ padding: "0.35rem 0.4rem", textAlign: "right", fontFamily: "monospace", color: extVal === "—" ? "var(--text-secondary)" : "#6366f1" }}>{extVal}</td>
                              <td style={{ padding: "0.35rem 0.4rem", textAlign: "right", fontFamily: "monospace", color: rawVal === "—" ? "var(--text-secondary)" : "#f59e0b" }}>{rawVal}</td>
                              <td style={{ padding: "0.35rem 0.4rem", textAlign: "center" }}>{f.raw == null ? "—" : match ? "✅" : "⚠️"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </div>
          )}
          {rawSyncData.length === 0 && !rawSyncLoading && (
            <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-secondary)" }}>
              Select a user and click Inspect to view raw sync data.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

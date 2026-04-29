"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TierMeta {
  label: string;
  color: string;
  desc: string;
}

interface Thresholds {
  copula_min_days: number;
  ml_model_min_days: number;
  ml_lag2_min_days: number;
  nn_min_days: number;
  nn_accuracy_gain_pct: number;
  nn_consecutive_wins: number;
}

interface MaturityStatus {
  data_days: number;
  active_tier: string;
  active_tier_label: string;
  active_tier_color: string;
  active_tier_desc: string;
  next_tier: string | null;
  next_tier_label: string | null;
  prev_tier: string | null;
  prev_tier_label: string | null;
  qualifies_for_next: boolean;
  recommendation_text: string;
  can_advance: boolean;
  can_revert: boolean;
  thresholds: Thresholds;
  tier_meta: Record<string, TierMeta>;
  nn_winning: boolean;
}

interface FeatureImportance {
  name: string;
  importance: number;
}

interface OutcomeCard {
  outcome_name: string;
  outcome_label: string;
  r_squared: number;
  r_squared_train: number;
  n_samples: number;
  top_features: FeatureImportance[];
  explanation: string;
  low_r2: boolean;
  suggestion: string;
}

interface EvalsPayload {
  available: boolean;
  reason?: string;
  fitted_at?: string;
  data_days?: number;
  avg_r_squared?: number;
  include_rolling?: boolean;
  outcome_cards?: OutcomeCard[];
  trajectory?: TrajectoryEntry[];
  maturity?: MaturityStatus;
}

interface TrajectoryEntry {
  fitted_at: string;
  data_days: number;
  r_squared: Record<string, number>;
}

interface Props {
  userId: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const R2_GREEN = 0.6;
const R2_AMBER = 0.3;

function r2Color(r2: number): string {
  if (r2 >= R2_GREEN) return "#22c55e";
  if (r2 >= R2_AMBER) return "#f59e0b";
  return "#ef4444";
}

function r2Label(r2: number): string {
  if (r2 >= R2_GREEN) return "Strong";
  if (r2 >= R2_AMBER) return "Moderate";
  return "Weak";
}

function featureReadable(name: string): string {
  const map: Record<string, string> = {
    sleep_duration_hours: "Sleep Duration",
    bedtime_hour_cos: "Bedtime Timing",
    bedtime_hour_sin: "Bedtime Timing",
    active_minutes_h: "Active Minutes",
    exercise_type_idx: "Exercise Type",
    exercise_duration_h: "Exercise Duration",
    steps_1000s: "Daily Steps",
    prev_delta_sleep_score: "Prev Sleep Score Δ",
    prev_delta_hrv_ms: "Prev HRV Δ",
    prev_delta_rhr_bpm: "Prev Resting HR Δ",
    prev_delta_stress: "Prev Stress Δ",
    prev_delta_body_battery: "Prev Body Battery Δ",
    prev_delta_sleep_stage_quality: "Prev Sleep Quality Δ",
    prev_delta_vo2_max: "Prev VO2 Max Δ",
    compliance_sleep: "Sleep Compliance",
    compliance_activity: "Activity Compliance",
  };
  const base = name.replace("_roll7_mean", " (7d avg)").replace("_roll7_std", " (7d var)");
  const baseKey = name.replace(/_roll7_(mean|std)$/, "");
  return map[baseKey] ? map[baseKey] + (name !== baseKey ? name.replace(baseKey, "") : "") : base.replace(/_/g, " ");
}

const TIER_ORDER_LIST = ["rules", "copula", "ml_model", "nn"];

function getGateDays(tier: string, thresholds: Thresholds): number {
  const map: Record<string, number> = {
    rules: 0,
    copula: thresholds.copula_min_days,
    ml_model: thresholds.ml_model_min_days,
    nn: thresholds.nn_min_days,
  };
  return map[tier] ?? 0;
}

type TierState =
  | "current"
  | "past"
  | "next-available"
  | "next-pending-calibrate"
  | "next-pending-days"
  | "future";

function getTierState(
  tier: string,
  idx: number,
  activeIdx: number,
  maturity: MaturityStatus,
): TierState {
  if (idx === activeIdx) return "current";
  if (idx < activeIdx) return "past";
  if (idx === activeIdx + 1) {
    if (maturity.can_advance) return "next-available";
    const gateDays = getGateDays(tier, maturity.thresholds);
    if (maturity.data_days >= gateDays) return "next-pending-calibrate";
    return "next-pending-days";
  }
  return "future";
}

function TierPipeline({
  maturity,
  advancing,
  jumping,
  calibrating,
  onAdvance,
  onJumpTo,
  onCalibrate,
}: {
  maturity: MaturityStatus;
  advancing: boolean;
  jumping: boolean;
  calibrating: boolean;
  onAdvance: () => void;
  onJumpTo: (tier: string) => void;
  onCalibrate: () => void;
}) {
  const activeIdx = TIER_ORDER_LIST.indexOf(maturity.active_tier);

  // Build connector progress: fraction of the way through the pipeline
  const progressPct = activeIdx / (TIER_ORDER_LIST.length - 1);

  return (
    <div style={{ overflowX: "auto", paddingBottom: "0.25rem" }}>
      <div style={{ position: "relative", padding: "0.5rem 0 1.5rem", minWidth: 560 }}>
        {/* ── Track (full gray) ── */}
        <div
          style={{
            position: "absolute",
            top: "calc(0.5rem + 20px)",
            left: "calc(12.5%)",
            right: "calc(12.5%)",
            height: 2,
            background: "rgba(255,255,255,0.08)",
            borderRadius: 1,
          }}
        />
        {/* ── Track fill (completed portion) ── */}
        <div
          style={{
            position: "absolute",
            top: "calc(0.5rem + 20px)",
            left: "calc(12.5%)",
            width: `${progressPct * 75}%`,
            height: 2,
            background: maturity.tier_meta[maturity.active_tier]?.color ?? "#6366f1",
            borderRadius: 1,
            transition: "width 0.5s ease",
          }}
        />

        {/* ── Nodes ── */}
        <div style={{ display: "flex", justifyContent: "space-between", position: "relative", zIndex: 1 }}>
          {TIER_ORDER_LIST.map((tier, idx) => {
            const meta = maturity.tier_meta[tier];
            const state = getTierState(tier, idx, activeIdx, maturity);
            const gateDays = getGateDays(tier, maturity.thresholds);
            const daysRemaining = Math.max(0, gateDays - maturity.data_days);

            const isClickable =
              (state === "past" && !jumping) ||
              (state === "next-available" && !advancing) ||
              (state === "next-pending-calibrate" && !calibrating);

            const nodeColor =
              state === "current" || state === "past" || state === "next-available"
                ? meta.color
                : state === "next-pending-calibrate"
                ? "#f59e0b"
                : "#4b5563";

            const nodeBackground =
              state === "current"
                ? meta.color
                : state === "past"
                ? `${meta.color}33`
                : state === "next-available"
                ? `${meta.color}22`
                : state === "next-pending-calibrate"
                ? "rgba(245,158,11,0.12)"
                : "rgba(255,255,255,0.04)";

            const glowShadow =
              state === "current"
                ? `0 0 18px ${meta.color}88`
                : state === "next-available"
                ? `0 0 10px ${meta.color}55`
                : "none";

            const nodeIcon =
              state === "current"
                ? "✓"
                : state === "past"
                ? "↩"
                : state === "next-available"
                ? "→"
                : state === "next-pending-calibrate"
                ? "🧬"
                : state === "next-pending-days"
                ? String(daysRemaining) + "d"
                : "—";

            const statusText =
              state === "current" ? (
                <span style={{ color: meta.color, fontWeight: 700 }}>Active</span>
              ) : state === "past" ? (
                <span style={{ color: "#6b7280", cursor: "pointer" }}>Click to revert</span>
              ) : state === "next-available" ? (
                <span style={{ color: meta.color, fontWeight: 600 }}>Ready — click to unlock</span>
              ) : state === "next-pending-calibrate" ? (
                <span style={{ color: "#f59e0b" }}>Click to calibrate first</span>
              ) : state === "next-pending-days" ? (
                <span style={{ color: "var(--text-secondary, #888)" }}>{daysRemaining}d to unlock</span>
              ) : (
                <span style={{ color: "var(--text-secondary, #666)" }}>
                  {daysRemaining > 0 ? `${daysRemaining}d away` : "locked"}
                </span>
              );

            const handleClick = () => {
              if (!isClickable) return;
              if (state === "past") onJumpTo(tier);
              else if (state === "next-available") onAdvance();
              else if (state === "next-pending-calibrate") onCalibrate();
            };

            return (
              <div
                key={tier}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "0.4rem",
                  flex: 1,
                  opacity: state === "future" ? 0.38 : 1,
                  transition: "opacity 0.3s",
                }}
              >
                {/* Circle node */}
                <div
                  onClick={handleClick}
                  title={
                    state === "past"
                      ? `Revert to ${meta.label}`
                      : state === "next-available"
                      ? `Advance to ${meta.label}`
                      : state === "next-pending-calibrate"
                      ? "Run Deep Calibrate to unlock"
                      : meta.label
                  }
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: "50%",
                    background: nodeBackground,
                    border: `2px solid ${nodeColor}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: isClickable ? "pointer" : "default",
                    boxShadow: glowShadow,
                    fontSize: state === "next-pending-days" ? "0.6rem" : "0.95rem",
                    fontWeight: 700,
                    color: state === "current" ? "#fff" : nodeColor,
                    transition: "box-shadow 0.3s, background 0.3s",
                    userSelect: "none",
                  }}
                >
                  {nodeIcon}
                </div>

                {/* Tier label */}
                <div
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: state === "current" ? 700 : 600,
                    color:
                      state === "current"
                        ? meta.color
                        : state === "past"
                        ? "var(--text-primary, #fff)"
                        : "var(--text-secondary, #aaa)",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                  }}
                >
                  {meta.label}
                </div>

                {/* Gate requirement (days) */}
                {gateDays > 0 && (
                  <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", textAlign: "center" }}>
                    {gateDays}d gate
                  </div>
                )}

                {/* Status / action text */}
                <div style={{ fontSize: "0.7rem", textAlign: "center", lineHeight: 1.3, maxWidth: 100 }}>
                  {statusText}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  background: "var(--bg-card, #1e1e2e)",
  border: "1px solid var(--border-color, rgba(255,255,255,0.08))",
  borderRadius: "16px",
  padding: "1.5rem",
};

const btnStyle = (color: string, disabled: boolean): React.CSSProperties => ({
  padding: "0.5rem 1.25rem",
  borderRadius: "8px",
  border: "none",
  cursor: disabled ? "not-allowed" : "pointer",
  fontWeight: 600,
  fontSize: "0.875rem",
  background: disabled ? "rgba(255,255,255,0.05)" : color,
  color: disabled ? "var(--text-secondary, #888)" : "#fff",
  opacity: disabled ? 0.5 : 1,
  transition: "opacity 0.2s",
});

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TierBadge({ label, color }: { label: string; color: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.2rem 0.75rem",
        borderRadius: "999px",
        fontSize: "0.78rem",
        fontWeight: 700,
        background: `${color}22`,
        color,
        border: `1px solid ${color}55`,
      }}
    >
      {label}
    </span>
  );
}

function R2Badge({ r2 }: { r2: number }) {
  const color = r2Color(r2);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.3rem",
        padding: "0.15rem 0.6rem",
        borderRadius: "999px",
        fontSize: "0.75rem",
        fontWeight: 700,
        background: `${color}22`,
        color,
        border: `1px solid ${color}55`,
      }}
    >
      R² {(r2 * 100).toFixed(0)}% — {r2Label(r2)}
    </span>
  );
}

function FeatureBar({ name, importance }: { name: string; importance: number }) {
  return (
    <div style={{ marginBottom: "0.3rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-secondary, #aaa)", marginBottom: "0.15rem" }}>
        <span>{featureReadable(name)}</span>
        <span>{(importance * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 999, background: "rgba(255,255,255,0.07)" }}>
        <div
          style={{
            height: "100%",
            width: `${Math.min(100, importance * 100)}%`,
            borderRadius: 999,
            background: "var(--accent-primary, #6366f1)",
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

function OutcomeModelCard({ card }: { card: OutcomeCard }) {
  return (
    <div
      style={{
        ...cardStyle,
        borderTop: `3px solid ${r2Color(card.r_squared)}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
        <h4 style={{ margin: 0, fontSize: "0.9rem", color: "var(--text-primary, #fff)" }}>{card.outcome_label}</h4>
        <R2Badge r2={card.r_squared} />
      </div>

      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary, #aaa)", margin: "0.5rem 0 0.75rem" }}>
        {card.explanation}
      </p>

      <div style={{ marginBottom: "0.5rem" }}>
        <div style={{ fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-secondary, #aaa)", marginBottom: "0.4rem" }}>
          Top predictors ({card.n_samples} samples)
        </div>
        {card.top_features.map((f) => (
          <FeatureBar key={f.name} name={f.name} importance={f.importance} />
        ))}
      </div>

      {card.low_r2 && card.suggestion && (
        <div
          style={{
            marginTop: "0.5rem",
            padding: "0.5rem 0.75rem",
            borderRadius: "8px",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.2)",
            fontSize: "0.75rem",
            color: "#fca5a5",
          }}
        >
          💡 {card.suggestion}
        </div>
      )}
    </div>
  );
}

function ThresholdEditor({
  thresholds,
  onSave,
}: {
  thresholds: Thresholds;
  onSave: (overrides: Partial<Thresholds>) => void;
}) {
  const [local, setLocal] = useState({ ...thresholds });

  const fields: { key: keyof Thresholds; label: string; step: number }[] = [
    { key: "copula_min_days", label: "Copula Gate (days)", step: 1 },
    { key: "ml_model_min_days", label: "ML Model Gate (days)", step: 1 },
    { key: "ml_lag2_min_days", label: "Rolling Lag Gate (days)", step: 1 },
    { key: "nn_min_days", label: "Neural Net Gate (days)", step: 1 },
    { key: "nn_accuracy_gain_pct", label: "NN Accuracy Gain Threshold (%)", step: 0.5 },
    { key: "nn_consecutive_wins", label: "NN Consecutive Wins Required", step: 1 },
  ];

  return (
    <div style={{ marginTop: "1rem" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        {fields.map(({ key, label, step }) => (
          <label key={key} style={{ display: "flex", flexDirection: "column", gap: "0.2rem", fontSize: "0.8rem", color: "var(--text-secondary, #aaa)" }}>
            <span>{label}</span>
            <input
              type="number"
              step={step}
              value={local[key]}
              onChange={(e) =>
                setLocal((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))
              }
              style={{
                background: "var(--bg, #0f0f1a)",
                border: "1px solid var(--border-color, rgba(255,255,255,0.1))",
                borderRadius: "6px",
                padding: "0.4rem 0.6rem",
                color: "var(--text-primary, #fff)",
                fontSize: "0.85rem",
              }}
            />
          </label>
        ))}
      </div>
      <button
        onClick={() => onSave(local)}
        style={{ ...btnStyle("#6366f1", false), width: "100%" }}
      >
        Save Threshold Overrides
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory chart
// ---------------------------------------------------------------------------

const OUTCOME_COLORS: Record<string, string> = {
  delta_sleep_score: "#6366f1",
  delta_hrv_ms: "#22c55e",
  delta_rhr_bpm: "#ef4444",
  delta_stress: "#f59e0b",
  delta_body_battery: "#3b82f6",
  delta_sleep_stage_quality: "#a855f7",
  delta_vo2_max: "#14b8a6",
};

const OUTCOME_SHORT: Record<string, string> = {
  delta_sleep_score: "Sleep",
  delta_hrv_ms: "HRV",
  delta_rhr_bpm: "RHR",
  delta_stress: "Stress",
  delta_body_battery: "Battery",
  delta_sleep_stage_quality: "Sleep Q",
  delta_vo2_max: "VO2",
};

function TrajectoryChart({ trajectory }: { trajectory: TrajectoryEntry[] }) {
  if (!trajectory || trajectory.length === 0) {
    return (
      <p style={{ color: "var(--text-secondary, #aaa)", fontSize: "0.85rem", textAlign: "center" }}>
        No training history yet. Train again after collecting more data to see R² trends.
      </p>
    );
  }

  const data = trajectory.map((entry) => ({
    date: entry.fitted_at.slice(0, 10),
    days: entry.data_days,
    ...Object.fromEntries(
      Object.entries(entry.r_squared).map(([k, v]) => [OUTCOME_SHORT[k] || k, Math.round(v * 100)])
    ),
  }));

  const outcomeKeys = Object.keys(trajectory[0].r_squared);

  return (
    <div style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="date" fontSize={10} stroke="var(--text-secondary, #aaa)" />
          <YAxis tickFormatter={(v: number) => `${v}%`} domain={[-10, 100]} fontSize={10} stroke="var(--text-secondary, #aaa)" />
          <Tooltip
            contentStyle={{ background: "var(--bg, #0f0f1a)", border: "1px solid var(--border-color)", borderRadius: 10, fontSize: "0.8rem" }}
            formatter={(v: number, name: string) => [`${v}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: "0.75rem", paddingTop: 8 }} />
          {outcomeKeys.map((k) => (
            <Line
              key={k}
              type="monotone"
              dataKey={OUTCOME_SHORT[k] || k}
              stroke={OUTCOME_COLORS[k] || "#888"}
              strokeWidth={2}
              dot={data.length <= 8}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function OutcomeModelEvals({ userId, onRecalculate }: Props & { onRecalculate?: () => void }) {
  const [evals, setEvals] = useState<EvalsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [advancing, setAdvancing] = useState(false);
  const [jumping, setJumping] = useState(false);
  const [training, setTraining] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [recalculating, setRecalculating] = useState(false);
  const [showThresholds, setShowThresholds] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const fetchEvals = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/evals/models", {
        headers: { "x-user-id": String(userId) },
      });
      const data = await res.json();
      setEvals(data);
    } catch (err) {
      setEvals({ available: false, reason: "Failed to load evals." });
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchEvals();
  }, [fetchEvals]);

  const showFeedback = (msg: string) => {
    setFeedback(msg);
    setTimeout(() => setFeedback(null), 3500);
  };

  const handleAdvance = async () => {
    setAdvancing(true);
    try {
      const res = await fetch("/api/maturity/transition", {
        method: "POST",
        headers: { "x-user-id": String(userId) },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Advance failed");
      showFeedback(`Tier advanced to ${data.active_tier}`);
      await fetchEvals();
    } catch (err: any) {
      showFeedback(`Error: ${err.message}`);
    } finally {
      setAdvancing(false);
    }
  };

  const handleJumpTo = async (tier: string) => {
    setJumping(true);
    try {
      const res = await fetch("/api/maturity/jump", {
        method: "POST",
        headers: { "x-user-id": String(userId), "content-type": "application/json" },
        body: JSON.stringify({ tier }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Jump failed");
      const label = evals?.maturity?.tier_meta?.[tier]?.label ?? tier;
      showFeedback(`Reverted to ${label}`);
      await fetchEvals();
    } catch (err: any) {
      showFeedback(`Error: ${err.message}`);
    } finally {
      setJumping(false);
    }
  };

  const handleCalibrate = async () => {
    setCalibrating(true);
    try {
      const res = await fetch("/api/calibrate", {
        method: "POST",
        headers: { "x-user-id": String(userId) },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Calibration failed");
      const r2 = data.r_squared != null ? ` R²=${(data.r_squared * 100).toFixed(0)}%` : "";
      showFeedback(`Deep calibration done.${r2} You can now Advance → Copula.`);
      await fetchEvals();
    } catch (err: any) {
      showFeedback(`Calibration error: ${err.message}`);
    } finally {
      setCalibrating(false);
    }
  };

  const handleRecalculate = async () => {
    setRecalculating(true);
    try {
      const res = await fetch("/api/persona/evals/recalculate", {
        method: "POST",
        headers: { "x-user-id": String(userId) },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Recalculate failed");
      showFeedback(`Recalculated ${data.records_updated} eval records — compliance is now backfilled.`);
      await fetchEvals();
      onRecalculate?.();
    } catch (err: any) {
      showFeedback(`Error: ${err.message}`);
    } finally {
      setRecalculating(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    try {
      const res = await fetch("/api/maturity/train", {
        method: "POST",
        headers: { "x-user-id": String(userId) },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Training failed");
      showFeedback(`Trained ${data.n_models} outcome models on ${data.data_days} days`);
      await fetchEvals();
    } catch (err: any) {
      showFeedback(`Error: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  const handleSaveThresholds = async (overrides: Partial<Thresholds>) => {
    try {
      const res = await fetch("/api/maturity/thresholds", {
        method: "PUT",
        headers: { "x-user-id": String(userId), "content-type": "application/json" },
        body: JSON.stringify(overrides),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Save failed");
      showFeedback("Thresholds saved.");
      await fetchEvals();
    } catch (err: any) {
      showFeedback(`Error: ${err.message}`);
    }
  };

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", padding: "3rem", color: "var(--text-secondary, #aaa)" }}>
        Loading model evals…
      </div>
    );
  }

  const maturity = evals?.maturity;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Feedback toast */}
      {feedback && (
        <div
          style={{
            position: "fixed",
            bottom: "1.5rem",
            right: "1.5rem",
            background: "#1e293b",
            border: "1px solid var(--border-color)",
            borderRadius: 12,
            padding: "0.75rem 1.25rem",
            fontSize: "0.875rem",
            color: "#fff",
            zIndex: 9999,
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
          }}
        >
          {feedback}
        </div>
      )}

      {/* ── Section A: Tier Pipeline ──────────────────────────────────── */}
      {maturity && (
        <div style={cardStyle}>
          {/* Header row */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem" }}>
            <div>
              <div style={{ fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-secondary, #aaa)", marginBottom: "0.3rem" }}>
                Model Maturity Pipeline
              </div>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <TierBadge label={maturity.active_tier_label} color={maturity.active_tier_color} />
                <span style={{ fontSize: "0.8rem", color: "var(--text-secondary, #aaa)" }}>
                  {maturity.data_days} paired days
                </span>
              </div>
              <p style={{ margin: "0.4rem 0 0", fontSize: "0.82rem", color: "var(--text-secondary, #aaa)" }}>
                {maturity.recommendation_text}
              </p>
            </div>
          </div>

          {/* Visual tier pipeline */}
          <TierPipeline
            maturity={maturity}
            advancing={advancing}
            jumping={jumping}
            calibrating={calibrating}
            onAdvance={handleAdvance}
            onJumpTo={handleJumpTo}
            onCalibrate={handleCalibrate}
          />

          {/* Secondary actions */}
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <button
              onClick={handleTrain}
              disabled={training}
              style={btnStyle("#22c55e", training)}
            >
              {training ? "Training…" : "Re-train Models"}
            </button>

            <button
              onClick={handleRecalculate}
              disabled={recalculating}
              style={btnStyle("#0ea5e9", recalculating)}
            >
              {recalculating ? "Recalculating…" : "🔄 Recalculate Evals"}
            </button>

            <button
              onClick={() => setShowThresholds((v) => !v)}
              style={btnStyle("rgba(255,255,255,0.08)", false)}
            >
              {showThresholds ? "Hide Thresholds" : "Edit Thresholds"}
            </button>
          </div>

          {/* Threshold editor */}
          {showThresholds && (
            <ThresholdEditor
              thresholds={maturity.thresholds}
              onSave={handleSaveThresholds}
            />
          )}
        </div>
      )}

      {/* ── Section B: Outcome model cards (ml_model tier+) ────────────── */}
      {evals?.available && evals.outcome_cards && evals.outcome_cards.length > 0 && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: "1rem" }}>Per-Outcome Explainability</h3>
              <p style={{ margin: "0.25rem 0 0", fontSize: "0.8rem", color: "var(--text-secondary, #aaa)" }}>
                Leave-one-out cross-validated R² — how well each biomarker is explained by your inputs.
                {evals.include_rolling && " Includes 7-day rolling aggregates."}
              </p>
            </div>
            <span style={{ fontSize: "0.75rem", color: "var(--text-secondary, #aaa)" }}>
              Avg R²: {evals.avg_r_squared !== undefined ? (evals.avg_r_squared * 100).toFixed(0) + "%" : "—"}
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "1rem",
            }}
          >
            {evals.outcome_cards.map((card) => (
              <OutcomeModelCard key={card.outcome_name} card={card} />
            ))}
          </div>
        </div>
      )}

      {/* ── ML models not yet available ─────────────────────────────────── */}
      {!evals?.available && (
        <div style={{ ...cardStyle, textAlign: "center", padding: "2.5rem" }}>
          <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>📈</div>
          <h4 style={{ margin: "0 0 0.5rem" }}>ML Models Not Yet Trained</h4>
          <p style={{ color: "var(--text-secondary, #aaa)", fontSize: "0.875rem", margin: "0 0 1rem" }}>
            {evals?.reason ?? `Collect ${maturity?.thresholds.ml_model_min_days ?? 30} paired days of data to unlock per-outcome Ridge regression models.`}
          </p>
          {maturity && (
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary, #aaa)" }}>
              You have{" "}
              <strong style={{ color: "var(--text-primary, #fff)" }}>{maturity.data_days}</strong>{" "}
              / {maturity.thresholds.ml_model_min_days} days needed.
            </div>
          )}
        </div>
      )}

      {/* ── Section E: R² trajectory chart ──────────────────────────────── */}
      {evals?.available && evals.trajectory && evals.trajectory.length > 0 && (
        <div style={cardStyle}>
          <h3 style={{ margin: "0 0 0.5rem" }}>R² Trend Over Time</h3>
          <p style={{ margin: "0 0 1rem", fontSize: "0.8rem", color: "var(--text-secondary, #aaa)" }}>
            How model explainability improves as more data is collected.
          </p>
          <TrajectoryChart trajectory={evals.trajectory} />
        </div>
      )}

      {/* ── Section D: Low-R² unexplained outcomes ───────────────────────── */}
      {evals?.available && evals.outcome_cards && evals.outcome_cards.some((c) => c.low_r2) && (
        <div style={{ ...cardStyle, borderColor: "rgba(239,68,68,0.25)" }}>
          <h3 style={{ margin: "0 0 0.5rem", color: "#fca5a5" }}>⚠ Outcomes Needing More Data</h3>
          <p style={{ margin: "0 0 1rem", fontSize: "0.8rem", color: "var(--text-secondary, #aaa)" }}>
            These biomarkers have R² below 30% — the model cannot reliably predict them yet.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {evals.outcome_cards
              .filter((c) => c.low_r2)
              .map((c) => (
                <div key={c.outcome_name} style={{ fontSize: "0.85rem" }}>
                  <strong style={{ color: "var(--text-primary, #fff)" }}>{c.outcome_label}</strong>
                  {c.suggestion && (
                    <span style={{ marginLeft: "0.5rem", color: "var(--text-secondary, #aaa)" }}>
                      — {c.suggestion}
                    </span>
                  )}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* ── Section F: NN vs ML comparison (nn tier) ─────────────────────── */}
      {maturity?.active_tier === "nn" && maturity.nn_winning && (
        <div style={{ ...cardStyle, borderColor: "rgba(34,197,94,0.3)" }}>
          <h3 style={{ margin: "0 0 0.5rem", color: "#86efac" }}>Neural Network Outperforming ML</h3>
          <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--text-secondary, #aaa)" }}>
            The NN has beaten the ML model accuracy threshold for the required number of consecutive
            retraining cycles. You are already on the optimal path. If you'd like to experiment,
            you can revert to the ML tier above.
          </p>
        </div>
      )}
    </div>
  );
}

import React from 'react';

export default function UserManual() {
  return (
    <div className="premium-card" style={{ padding: "2rem", lineHeight: 1.8 }}>
      <h2 style={{ fontSize: "1.75rem", marginBottom: "1.5rem", background: "linear-gradient(135deg, #fff 0%, #a5b4fc 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
        User Manual: AI Wellness Orchestrator
      </h2>
      
      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>1. Getting Started</h3>
        <p>Welcome to the Wellness Engine — your AI-powered digital twin that learns your body&apos;s unique response to sleep, exercise, and nutrition.</p>
        <ul style={{ paddingLeft: "1.25rem", color: "var(--text-secondary)" }}>
          <li>Go to the <strong>Settings</strong> tab and enter your <strong>Display Name</strong>.</li>
          <li>Connect your <strong>Garmin account</strong> by entering your email and password.</li>
          <li>Credentials are encrypted using <strong>AES-256</strong> before being stored.</li>
          <li>Once saved, Garmin data <strong>syncs automatically every 6 hours</strong> — no manual sync needed.</li>
        </ul>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>2. Choosing Your Goal</h3>
        <p>On the <strong>Dashboard</strong>, select your current health objective from the goal dropdown:</p>
        <ul style={{ paddingLeft: "1.25rem", color: "var(--text-secondary)" }}>
          <li>🎯 <strong>Overall Wellness</strong> — Balanced approach to all health markers</li>
          <li>⚖️ <strong>Reduce Weight</strong> — HIIT focus with high-protein nutrition</li>
          <li>💪 <strong>Increase Muscle</strong> — Strength training with recovery sleep</li>
          <li>💓 <strong>Better HRV</strong> — Recovery-focused training for nervous system balance</li>
          <li>🧘 <strong>Manage Stress</strong> — Mindful movement and anti-inflammatory nutrition</li>
          <li>🏆 <strong>Athletic Performance</strong> — Periodized intensity with optimal recovery</li>
        </ul>
        <p>Recommendations update automatically when you switch goals.</p>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>3. AI Coaching Fidelity Levels</h3>
        <p>The AI coach improves as more data accumulates. Your fidelity badge shows the current quality:</p>
        <div style={{ display: "grid", gap: "0.5rem", marginTop: "0.75rem" }}>
          {[
            { badge: "Generic", color: "#6b7280", desc: "No data yet — evidence-based defaults for your goal" },
            { badge: "Basic", color: "#f59e0b", desc: "Some Garmin/manual data — partially personalized" },
            { badge: "Calibrated", color: "#6366f1", desc: "Regression model fitted to your real data patterns" },
            { badge: "AI-Optimized", color: "#22c55e", desc: "Neural network trained on your digital twin — fully personalized" },
          ].map(level => (
            <div key={level.badge} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.5rem 0.75rem", background: `${level.color}08`, borderRadius: "var(--radius-sm)" }}>
              <span style={{ fontSize: "0.7rem", fontWeight: 700, padding: "0.1rem 0.5rem", borderRadius: "4px", background: `${level.color}20`, color: level.color, textTransform: "uppercase", minWidth: "80px", textAlign: "center" }}>{level.badge}</span>
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{level.desc}</span>
            </div>
          ))}
        </div>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>4. Logging Manual Data</h3>
        <p>On the <strong>Dashboard</strong>, scroll down to the manual log form. You can log data using natural language:</p>
        <code style={{ display: "block", padding: "1rem", background: "var(--bg-card)", borderRadius: "var(--radius-sm)", marginBottom: "0.5rem", fontSize: "0.85rem" }}>
          &quot;I had a huge protein bowl for lunch&quot; <br />
          &quot;I&apos;m feeling incredibly stressed today (Level 9)&quot; <br />
          &quot;My weight is 72kg today&quot;
        </code>
        <p>The AI parses these inputs and updates your digital twin&apos;s state. More manual logs improve recommendation accuracy.</p>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>5. Deep Calibration & NN Training (Admin)</h3>
        <p>Once you have <strong>7+ days of data</strong>, an admin can calibrate your digital twin from the <strong>🔒 Admin</strong> tab:</p>
        <div style={{ background: "rgba(99, 102, 241, 0.05)", padding: "1rem", borderRadius: "var(--radius-sm)", borderLeft: "4px solid var(--accent-primary)", marginBottom: "1rem" }}>
          <strong>Step 1: Deep Calibration</strong> — Runs a least-squares regression on your real data to discover how your body responds to sleep, exercise, stress, and nutrition. Shows the R² score for model quality.
        </div>
        <div style={{ background: "rgba(34, 197, 94, 0.05)", padding: "1rem", borderRadius: "var(--radius-sm)", borderLeft: "4px solid #22c55e" }}>
          <strong>Step 2: Train Neural Network</strong> — If the R² is acceptable, train a PPO reinforcement learning agent on your calibrated simulator. This creates a fully personalized AI coach (~30 seconds).
        </div>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h3 style={{ color: "var(--accent-primary)", fontSize: "1.1rem", marginBottom: "0.75rem" }}>6. Data We Track</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
          <div>
            <h4 style={{ fontSize: "0.9rem", color: "var(--accent-secondary)", marginBottom: "0.5rem" }}>From Garmin (Auto-Synced)</h4>
            <ul style={{ paddingLeft: "1.25rem", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
              <li>HRV (Heart Rate Variability)</li>
              <li>Resting Heart Rate</li>
              <li>Body Battery</li>
              <li>Intensity Minutes</li>
              <li>Active Calories</li>
              <li>Training Load</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: "0.9rem", color: "var(--accent-secondary)", marginBottom: "0.5rem" }}>From Manual Logs</h4>
            <ul style={{ paddingLeft: "1.25rem", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
              <li>Food / Calorie Intake</li>
              <li>Body Weight</li>
              <li>Stress Level (1-10)</li>
              <li>General Notes</li>
            </ul>
          </div>
        </div>
      </section>

      <footer style={{ marginTop: "3rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(255,255,255,0.1)", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
        Built for Advanced Wellness Optimization. Your Digital Twin is a mathematical representation of your physiological response to stress, nutrition, and exercise. As your data grows, the AI coach evolves from generic advice to fully personalized, neural-network-optimized coaching.
      </footer>
    </div>
  );
}

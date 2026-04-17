import GarminDashboard from "../components/GarminDashboard";

export default function Home() {
  return (
    <div style={{ padding: "clamp(1rem, 3vw, 4rem) clamp(0.75rem, 2vw, 2rem)", maxWidth: "1200px", margin: "0 auto", width: "100%" }}>
      <header style={{ marginBottom: "clamp(1rem, 3vw, 3rem)", textAlign: "center" }}>
        <h1 style={{ fontSize: "clamp(1.5rem, 5vw, 3rem)", marginBottom: "0.5rem", background: "linear-gradient(to right, var(--accent-primary), var(--accent-secondary))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Wellness Engine
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "clamp(0.85rem, 2vw, 1.2rem)" }}>
          Your digital twin dashboard and simulator calibration interface.
        </p>
      </header>
      
      <GarminDashboard />
    </div>
  );
}

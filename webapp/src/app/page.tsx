import GarminDashboard from "../components/GarminDashboard";

export default function Home() {
  return (
    <div style={{ padding: "4rem 2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <header style={{ marginBottom: "3rem", textAlign: "center" }}>
        <h1 style={{ fontSize: "3rem", marginBottom: "1rem", background: "linear-gradient(to right, var(--accent-primary), var(--accent-secondary))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Wellness Engine
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.2rem" }}>
          Your digital twin dashboard and simulator calibration interface.
        </p>
      </header>
      
      <GarminDashboard />
    </div>
  );
}

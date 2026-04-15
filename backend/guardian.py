import subprocess
import time
import sys
import os
import signal

def run_server():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")
    python_path = sys.executable

    print(f"[Guardian] Starting Wellness Server: {app_path}")
    
    while True:
        try:
            # Run the FastAPI app
            process = subprocess.Popen([python_path, app_path], 
                                      stdout=sys.stdout, 
                                      stderr=sys.stderr)
            
            # Wait for it to exit
            exit_code = process.wait()
            
            print(f"[Guardian] Server exited with code {exit_code}. Restarting in 2 seconds...")
            with open("guardian.log", "a") as f:
                f.write(f"{time.ctime()}: Server exited with code {exit_code}\n")
            
            time.sleep(2)
        except KeyboardInterrupt:
            print("[Guardian] Shutting down...")
            process.terminate()
            break
        except Exception as e:
            print(f"[Guardian] Error: {e}. Restarting...")
            time.sleep(5)

if __name__ == "__main__":
    run_server()

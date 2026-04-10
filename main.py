import subprocess
import sys

def main():
    print("Starting the Spotify Clustering Dashboard...")
    # Run the streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()

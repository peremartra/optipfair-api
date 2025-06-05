import os
import threading
import time
import uvicorn
from optipfair_backend import app as fastapi_app
from optipfair_frontend import create_interface

def run_fastapi():
    """Run FastAPI backend in a separate thread"""
    uvicorn.run(
        fastapi_app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )

def main():
    """Main function to start both FastAPI and Gradio"""
    
    # Start FastAPI in background thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Wait a moment for FastAPI to start
    print("🚀 Starting FastAPI backend...")
    time.sleep(3)
    
    # Create and launch Gradio interface
    print("🎨 Starting Gradio frontend...")
    interface = create_interface()
    
    # Launch configuration for HF Spaces
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
import subprocess
import time
import webbrowser
import sys
import os

def check_docker():
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Docker is installed and running.")
        else:
            print("Docker is not running. Please make sure Docker Desktop is started.")
            sys.exit(1)
    except FileNotFoundError:
        print("Docker is not installed. Please install Docker Desktop.")
        sys.exit(1)

def list_database_options():
    databases_dir = './databases'
    
    try:
        options = [f for f in os.listdir(databases_dir) if not f.startswith('.')]
        if not options:
            print("No collections found in the 'databases' folder.")
            sys.exit(1)
        
        print("Available collections in the 'databases' folder:")
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        
        return options
    except FileNotFoundError:
        print(f"The 'databases' folder does not exist at path: {databases_dir}")
        sys.exit(1)

def start_qdrant(selected_option):

    check_docker()

    try:
        print("Starting Qdrant Docker container...")
        
        abs_storage_path = os.path.abspath(f"./databases/{selected_option}")
        
        # Run the Docker container asynchronously. The script will continue running while the container is starting.
        process = subprocess.Popen(
            ["docker", "run", "-d", "-p", "6333:6333", "-v", f"{abs_storage_path}:/qdrant/storage", "qdrant/qdrant"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(10)

        # Check the process output
        stdout, stderr = process.communicate()
        print("Docker command output:")
        print(stdout)
        print(stderr)

        # Qdrant Web UI in the default web browser
        url = "http://localhost:6333/dashboard"
        print(f"Opening Qdrant Web UI at {url}")
        webbrowser.open(url)
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while starting Docker: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    options = list_database_options()

    selected_idx = int(input("Please enter the number of the collection you want to use: ")) - 1

    if selected_idx < 0 or selected_idx >= len(options):
        print("Invalid selection. Please run the script again and choose a valid option.")
        sys.exit(1)

    selected_option = options[selected_idx]

    start_qdrant(selected_option)


#!/bin/bash
# run: sudo sh setup.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "===== Startup script started at $(date) ====="

# Must run as root for apt and directory creation
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must be run as root"
    exit 1
fi

# Detect the actual user who called sudo
REAL_USER=${SUDO_USER:-$(whoami)}

# 1. Update system & Install packages
apt-get update -y
apt-get install -y python3 python3-venv python3-pip build-essential git nano

# 2. Setup Directory and Git Repo
TARGET_DIR="/nudge-x"
REPO_URL="https://github.com/realtechsupport/nudge-x.git"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating $TARGET_DIR and cloning repo..."
    mkdir -p "$TARGET_DIR"
    
    # Clone with no-checkout
    git clone --no-checkout "$REPO_URL" "$TARGET_DIR"
    
    # Move into the folder and bring the files into view
    cd "$TARGET_DIR"
    git checkout main  # Pulls the files so requirements.txt is visible
    
    # Ensure the real user owns the code
    chown -R "$REAL_USER":"$REAL_USER" "$TARGET_DIR"
fi

# 3. Virtual environment location (NOW INSIDE THE REPO)
VENV_DIR="$TARGET_DIR/venv"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    # Run as the REAL_USER so the venv is owned correctly from the start
    sudo -u "$REAL_USER" python3 -m venv "$VENV_DIR"
fi

# 4. Install Python Packages
echo "Installing dependencies from requirements.txt..."
# Check if requirements.txt exists before trying to install
if [ -f "$TARGET_DIR/requirements.txt" ]; then
    sudo -u "$REAL_USER" "$VENV_DIR/bin/python" -m pip install --upgrade pip
    sudo -u "$REAL_USER" "$VENV_DIR/bin/python" -m pip install numpy python-dotenv
fi

echo "===== Startup script completed at $(date) ====="
echo "To activate your venv, run: source $VENV_DIR/bin/activate"

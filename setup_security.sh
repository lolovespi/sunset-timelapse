#!/bin/bash
# Security Setup Script for Sunset Timelapse System
# This script secures file permissions and validates the security configuration

set -e  # Exit on error

echo "🔒 Sunset Timelapse Security Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running as correct user (not root)
if [ "$EUID" -eq 0 ]; then
    print_error "Do not run this script as root"
    exit 1
fi

# 1. Secure .env file
echo "1. Securing environment file..."
if [ -f ".env" ]; then
    chmod 600 .env
    print_status ".env permissions set to 600 (owner read/write only)"
else
    print_warning ".env file not found - create it from .env.example"
fi

# 2. Secure credentials directory
echo ""
echo "2. Securing credentials directory..."
CREDS_DIR="$HOME/.config/sunset-timelapse/credentials"
if [ -d "$CREDS_DIR" ]; then
    chmod 700 "$CREDS_DIR"
    chmod 600 "$CREDS_DIR"/* 2>/dev/null || true
    print_status "Credentials directory secured: $CREDS_DIR"
else
    mkdir -p "$CREDS_DIR"
    chmod 700 "$CREDS_DIR"
    print_status "Created secure credentials directory: $CREDS_DIR"
fi

# 3. Secure data directories
echo ""
echo "3. Securing data directories..."
if [ -d "data" ]; then
    chmod 700 data
    chmod 700 data/logs 2>/dev/null || true
    chmod 700 data/images 2>/dev/null || true
    chmod 700 data/videos 2>/dev/null || true
    print_status "Data directories secured (700)"
fi

# 4. Check for sensitive files in git
echo ""
echo "4. Checking for sensitive files..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Check if any sensitive files are tracked
    SENSITIVE_FILES=$(git ls-files | grep -E '\.(env|token|credentials|secret)' || true)
    if [ -z "$SENSITIVE_FILES" ]; then
        print_status "No sensitive files tracked in git"
    else
        print_error "Sensitive files found in git:"
        echo "$SENSITIVE_FILES"
        echo "Remove them with: git rm --cached <file>"
    fi
fi

# 5. Validate Python dependencies for security
echo ""
echo "5. Checking Python dependencies..."
if command -v pip3 &> /dev/null; then
    if pip3 list --format=freeze | grep -q "PyYAML\|requests\|Pillow"; then
        print_status "Core dependencies installed"
    else
        print_warning "Some dependencies may be missing - run: pip install -r requirements.txt"
    fi
fi

# 6. Check file permissions summary
echo ""
echo "6. Security Summary:"
echo "===================="
ls -la .env 2>/dev/null | tail -1 || print_warning ".env not found"
echo ""
echo "Credentials directory:"
ls -la "$CREDS_DIR" 2>/dev/null | head -5 || print_warning "No credentials found"

echo ""
echo "=================================="
print_status "Security setup complete!"
echo ""
echo "Next steps:"
echo "  1. Ensure .env file contains all required secrets"
echo "  2. Never commit .env or credential files to git"
echo "  3. Run 'python main.py config --validate' to test configuration"
echo "  4. Review SECURITY.md for additional hardening steps"

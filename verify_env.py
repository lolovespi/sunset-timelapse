#!/usr/bin/env python3
"""
Environment Verification Script
Checks if your .env file is properly configured
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

def check_env_file():
    """Check if .env file exists and is readable"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Create it from .env.example: cp .env.example .env")
        return False
        
    try:
        load_dotenv()
        print("✅ .env file found and loaded")
        return True
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False

def check_camera_password():
    """Check camera password"""
    password = os.getenv('CAMERA_PASSWORD')
    
    if not password:
        print("❌ CAMERA_PASSWORD not set in .env")
        print("   Add: CAMERA_PASSWORD=your_camera_password")
        return False
    
    if password == "your_camera_password" or password == "your_actual_camera_password":
        print("❌ CAMERA_PASSWORD still has placeholder value")
        print("   Update with your real camera password")
        return False
        
    print("✅ CAMERA_PASSWORD is set")
    return True

def check_google_credentials():
    """Check Google credentials file"""
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        print("   Add: GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json")
        return False
        
    creds_file = Path(creds_path)
    
    if not creds_file.exists():
        print(f"❌ Google credentials file not found: {creds_path}")
        print("   Download credentials from Google Cloud Console")
        return False
        
    try:
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
            
        # Check if it's OAuth credentials
        if 'client_id' in creds_data and 'client_secret' in creds_data:
            print("✅ Google OAuth credentials file found and valid")
            return True
            
        # Check if it's service account credentials
        elif 'private_key' in creds_data and 'client_email' in creds_data:
            print("✅ Google service account credentials file found and valid")
            print("   Note: You may need OAuth credentials for YouTube uploads")
            return True
            
        else:
            print("❌ Google credentials file format not recognized")
            print("   Should be OAuth client secrets or service account JSON")
            return False
            
    except json.JSONDecodeError:
        print(f"❌ Google credentials file is not valid JSON: {creds_path}")
        return False
    except Exception as e:
        print(f"❌ Error reading Google credentials file: {e}")
        return False

def check_file_permissions():
    """Check file permissions for security"""
    env_path = Path('.env')
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    issues = []
    
    # Check .env permissions
    if env_path.exists():
        stat = env_path.stat()
        mode = stat.st_mode & 0o777
        if mode != 0o600:
            issues.append(f".env has permissions {oct(mode)}, should be 0o600")
            
    # Check credentials permissions
    if creds_path and Path(creds_path).exists():
        stat = Path(creds_path).stat()
        mode = stat.st_mode & 0o777
        if mode > 0o600:
            issues.append(f"Credentials file has permissions {oct(mode)}, should be 0o600 or less")
            
    if issues:
        print("⚠️  File permission issues:")
        for issue in issues:
            print(f"   {issue}")
        print("   Fix with: chmod 600 .env google-credentials.json")
        return False
    else:
        print("✅ File permissions are secure")
        return True

def main():
    """Main verification function"""
    print("🔍 Verifying environment configuration...\n")
    
    checks = [
        check_env_file(),
        check_camera_password(),
        check_google_credentials(),
        check_file_permissions()
    ]
    
    print(f"\n📊 Results: {sum(checks)}/{len(checks)} checks passed")
    
    if all(checks):
        print("\n🎉 Environment is properly configured!")
        print("   You can now run: python main.py test")
    else:
        print("\n❌ Environment needs configuration")
        print("   Fix the issues above and run this script again")
        
        print("\n📚 Quick fixes:")
        print("   1. Copy template: cp .env.example .env")
        print("   2. Edit .env: nano .env")
        print("   3. Add camera password and Google credentials path")
        print("   4. Download Google credentials from Cloud Console")
        print("   5. Set secure permissions: chmod 600 .env *.json")

if __name__ == '__main__':
    main()

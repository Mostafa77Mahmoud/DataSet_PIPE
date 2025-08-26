#!/usr/bin/env python3
"""
Test script to verify Google Docs access and provide alternative solutions
"""

import requests
import sys

def test_docs_access():
    """Test access to the provided Google Docs URLs"""
    
    urls = {
        "Arabic": "https://docs.google.com/document/d/120Uhx6oNHPSYJlT7USekWjvoVZ1iQ1BP/edit?usp=sharing&ouid=113300385168711008791&rtpof=true&sd=true",
        "English": "https://docs.google.com/document/d/1jvUmwQv6DFOvwA5nT5XIANEwrsp7_IHO/edit?usp=drive_link&ouid=113300385168711008791&rtpof=true&sd=true"
    }
    
    export_urls = {
        "Arabic": "https://docs.google.com/document/d/120Uhx6oNHPSYJlT7USekWjvoVZ1iQ1BP/export?format=html",
        "English": "https://docs.google.com/document/d/1jvUmwQv6DFOvwA5nT5XIANEwrsp7_IHO/export?format=html"
    }
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    print("Testing Google Docs Access:")
    print("=" * 50)
    
    for lang, url in export_urls.items():
        print(f"\n{lang} Document:")
        try:
            response = session.head(url, timeout=30)
            print(f"  Status Code: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print(f"  ✓ Access granted")
            elif response.status_code == 432:
                print(f"  ✗ Access denied - Document may need public sharing")
            else:
                print(f"  ? Unexpected status code")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Potential Solutions:")
    print("1. Make documents publicly viewable (Share > Anyone with link can view)")
    print("2. Use Google Drive API with authentication")
    print("3. Download documents manually and provide local file paths")
    print("4. Export as plain text/PDF and provide alternative URLs")

if __name__ == "__main__":
    test_docs_access()
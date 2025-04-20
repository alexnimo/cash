import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        # Check if ffmpeg is installed
        try:
            subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("ffmpeg is already installed.")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("ffmpeg is not installed.")
            
            if sys.platform.startswith('win'):
                print("WARNING: On Windows, ffmpeg must be installed manually:")
                print("  - Download from https://ffmpeg.org/download.html")
                print("  - Add the ffmpeg bin folder to your PATH")
                print("\nAfter installing ffmpeg, you may need to restart your terminal.")
            elif sys.platform.startswith('darwin'):  # macOS
                print("Installing ffmpeg using Homebrew...")
                try:
                    # Check if Homebrew is installed
                    subprocess.check_call(['brew', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Install ffmpeg
                    subprocess.check_call(['brew', 'install', 'ffmpeg'])
                    print("ffmpeg installed successfully.")
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("ERROR: Homebrew is not installed. Please install Homebrew first:")
                    print("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                    print("Then run the setup again.")
            else:  # Linux
                print("Installing ffmpeg using apt...")
                try:
                    # Try using apt
                    subprocess.check_call(['sudo', 'apt', 'update'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.check_call(['sudo', 'apt', 'install', '-y', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("ffmpeg installed successfully.")
                except (subprocess.SubprocessError, FileNotFoundError):
                    try:
                        # Try using yum if apt fails
                        subprocess.check_call(['sudo', 'yum', 'install', '-y', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print("ffmpeg installed successfully.")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("ERROR: Could not install ffmpeg automatically.")
                        print("Please install manually:")
                        print("  - Using apt: sudo apt update && sudo apt install ffmpeg")
                        print("  - Using yum: sudo yum install ffmpeg")
        
        # Proceed with normal installation
        install.run(self)

setup(
    name="video-analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    cmdclass={
        'install': CustomInstall,
    },
)

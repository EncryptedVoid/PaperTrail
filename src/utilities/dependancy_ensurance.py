import platform
import shutil
import subprocess
import sys
from typing import Tuple


def _run_command(cmd: list, timeout: int = 30, shell: bool = False) -> Tuple[bool, str]:
    """Helper function to run shell commands safely."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=shell,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def _get_os() -> str:
    """Get the current operating system."""
    return platform.system()


def _ensure_wsl() -> bool:
    """Ensure WSL is available on Windows. Returns True if available."""
    if _get_os() != "Windows":
        return False

    # Check if wsl command exists
    if shutil.which("wsl"):
        # Check if a distro is installed
        success, output = _run_command(["wsl", "--list", "--quiet"])
        if success and output.strip():
            print("✓ WSL is available")
            return True

    print("⚠ WSL not found. Install with: wsl --install")
    return False


def _run_in_wsl(command: str, timeout: int = 300) -> Tuple[bool, str]:
    """Run a command in WSL."""
    wsl_cmd = ["wsl", "bash", "-c", command]
    return _run_command(wsl_cmd, timeout=timeout)


def _clean_choco_lock_files():
    """Clean up Chocolatey lock files that cause issues."""
    try:
        lock_patterns = [
            "C:\\ProgramData\\chocolatey\\lib\\*lock*",
            "C:\\ProgramData\\chocolatey\\lib-bad",
        ]
        for pattern in lock_patterns:
            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"Remove-Item -Force -Recurse '{pattern}' -ErrorAction SilentlyContinue",
                ],
                capture_output=True,
                timeout=10,
            )
    except:
        pass


def _install_via_choco(package: str, timeout: int = 300) -> bool:
    """Install package via Chocolatey with automatic cleanup and fallback."""
    if not shutil.which("choco"):
        return False

    # Clean lock files first
    _clean_choco_lock_files()

    try:
        # Try normal install with force and yes to all
        result = subprocess.run(
            ["choco", "install", "-y", "--force", "--ignore-checksums", package],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            input="Y\n",  # Auto-answer yes
        )

        if result.returncode == 0:
            return True

        # If failed, clean again and try elevated
        _clean_choco_lock_files()

        # Try with PowerShell elevation (will prompt for admin)
        ps_cmd = f'Start-Process choco -ArgumentList "install -y --force --ignore-checksums {package}" -Verb RunAs -Wait -WindowStyle Hidden'
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0:
            return True

        return False
    except Exception as e:
        print(f"⚠ Chocolatey error: {e}")
        return False


def _install_via_winget(package_id: str, timeout: int = 300) -> bool:
    """Install package via winget."""
    if not shutil.which("winget"):
        return False

    try:
        result = subprocess.run(
            [
                "winget",
                "install",
                "--id",
                package_id,
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_ffmpeg(min_version: str = "4.0") -> str:
    """Ensure FFmpeg is installed with WSL fallback."""
    # Check if already installed
    if shutil.which("ffmpeg"):
        success, output = _run_command(["ffmpeg", "-version"])
        if success:
            version_line = output.split("\n")[0]
            print(f"✓ FFmpeg already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing FFmpeg on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "ffmpeg"], check=False, timeout=300
            )

        elif os_type == "Windows":
            # Try winget first
            if _install_via_winget("Gyan.FFmpeg"):
                print("✓ Installed via winget")
            # Try chocolatey
            elif _install_via_choco("ffmpeg"):
                print("✓ Installed via chocolatey")
            # Fallback to WSL
            else:
                print("⚠ Native install failed, using WSL...")
                if _ensure_wsl():
                    print("Installing in WSL...")
                    _run_in_wsl("sudo apt update -y && sudo apt install -y ffmpeg")
                    success, output = _run_in_wsl("ffmpeg -version")
                    if success:
                        print("✓ FFmpeg installed in WSL")
                        return "ffmpeg (WSL)"
                    else:
                        print("⚠ WSL install failed, ffmpeg may not work")
                        return "ffmpeg (unavailable)"
                else:
                    print("⚠ WSL unavailable, ffmpeg not installed")
                    return "ffmpeg (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=False, timeout=300)

        # Verify installation
        if shutil.which("ffmpeg"):
            success, output = _run_command(["ffmpeg", "-version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ FFmpeg installed: {version_line}")
                return version_line

        # Check WSL as last resort
        if os_type == "Windows":
            success, output = _run_in_wsl("ffmpeg -version")
            if success:
                print("✓ FFmpeg available in WSL")
                return "ffmpeg (WSL)"

        print("⚠ FFmpeg installation incomplete, may not work")
        return "ffmpeg (unavailable)"

    except Exception as e:
        print(f"⚠ FFmpeg installation error: {e}")
        return "ffmpeg (unavailable)"


def ensure_imagemagick(min_version: str = "6.9") -> str:
    """Ensure ImageMagick is installed with WSL fallback."""
    # Check if already installed
    if shutil.which("convert") or shutil.which("magick"):
        cmd = "magick" if shutil.which("magick") else "convert"
        success, output = _run_command([cmd, "-version"])
        if success and "ImageMagick" in output:
            version_line = output.split("\n")[0]
            print(f"✓ ImageMagick already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing ImageMagick on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "imagemagick"],
                check=False,
                timeout=300,
            )

        elif os_type == "Windows":
            if _install_via_winget("ImageMagick.ImageMagick"):
                print("✓ Installed via winget")
            elif _install_via_choco("imagemagick"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Native install failed, using WSL...")
                if _ensure_wsl():
                    _run_in_wsl("sudo apt update -y && sudo apt install -y imagemagick")
                    success, output = _run_in_wsl("convert -version")
                    if success:
                        print("✓ ImageMagick installed in WSL")
                        return "imagemagick (WSL)"
                print("⚠ ImageMagick not fully installed")
                return "imagemagick (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", "imagemagick"], check=False, timeout=300
                )

        # Verify
        if shutil.which("convert") or shutil.which("magick"):
            cmd = "magick" if shutil.which("magick") else "convert"
            success, output = _run_command([cmd, "-version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ ImageMagick installed: {version_line}")
                return version_line

        if os_type == "Windows":
            success, output = _run_in_wsl("convert -version")
            if success:
                print("✓ ImageMagick available in WSL")
                return "imagemagick (WSL)"

        return "imagemagick (unavailable)"

    except Exception as e:
        print(f"⚠ ImageMagick error: {e}")
        return "imagemagick (unavailable)"


def ensure_pandoc(min_version: str = "2.0") -> str:
    """Ensure Pandoc is installed with WSL fallback."""
    if shutil.which("pandoc"):
        success, output = _run_command(["pandoc", "--version"])
        if success:
            version_line = output.split("\n")[0]
            print(f"✓ Pandoc already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing Pandoc on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "pandoc"], check=False, timeout=300
            )

        elif os_type == "Windows":
            if _install_via_winget("JohnMacFarlane.Pandoc"):
                print("✓ Installed via winget")
            elif _install_via_choco("pandoc"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Native install failed, using WSL...")
                if _ensure_wsl():
                    _run_in_wsl("sudo apt update -y && sudo apt install -y pandoc")
                    success, output = _run_in_wsl("pandoc --version")
                    if success:
                        print("✓ Pandoc installed in WSL")
                        return "pandoc (WSL)"
                return "pandoc (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "pandoc"], check=False, timeout=300)

        if shutil.which("pandoc"):
            success, output = _run_command(["pandoc", "--version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ Pandoc installed: {version_line}")
                return version_line

        if os_type == "Windows":
            success, output = _run_in_wsl("pandoc --version")
            if success:
                print("✓ Pandoc available in WSL")
                return "pandoc (WSL)"

        return "pandoc (unavailable)"

    except Exception as e:
        print(f"⚠ Pandoc error: {e}")
        return "pandoc (unavailable)"


def ensure_unpaper(min_version: str = "6.1") -> str:
    """Ensure unpaper is installed with WSL fallback. NEVER crashes."""
    # Check if already installed
    if shutil.which("unpaper"):
        success, output = _run_command(["unpaper", "--version"])
        if success:
            version_line = output.strip().split("\n")[0]
            print(f"✓ unpaper already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing unpaper on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "unpaper"], check=False, timeout=300
            )

        elif os_type == "Windows":
            # Try chocolatey (will likely fail due to your permissions)
            if _install_via_choco("unpaper"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Chocolatey failed (expected), falling back to WSL...")
                if _ensure_wsl():
                    print("Installing unpaper in WSL...")
                    _run_in_wsl("sudo apt update -y && sudo apt install -y unpaper")
                    success, output = _run_in_wsl("unpaper --version")
                    if success:
                        print("✓ unpaper installed in WSL")
                        return "unpaper (WSL)"
                    else:
                        print("⚠ WSL install incomplete")
                else:
                    print("⚠ WSL unavailable. unpaper will not be available.")
                    print("   Document cleaning features will be limited.")
                return "unpaper (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "unpaper"], check=False, timeout=300)

        # Verify
        if shutil.which("unpaper"):
            success, output = _run_command(["unpaper", "--version"])
            if success:
                version_line = output.strip().split("\n")[0]
                print(f"✓ unpaper installed: {version_line}")
                return version_line

        # Check WSL
        if os_type == "Windows":
            success, output = _run_in_wsl("unpaper --version")
            if success:
                print("✓ unpaper available in WSL")
                return "unpaper (WSL)"

        print("⚠ unpaper not installed, features will be limited")
        return "unpaper (unavailable)"

    except Exception as e:
        print(f"⚠ unpaper installation issue: {e}")
        print("   Document cleaning will be limited")
        return "unpaper (unavailable)"


def ensure_par2(min_version: str = "0.8") -> str:
    """Ensure par2 is installed with WSL fallback."""
    if shutil.which("par2"):
        success, output = _run_command(["par2", "-V"])
        if success:
            version_line = output.split("\n")[0] if output else "par2 (version unknown)"
            print(f"✓ par2 already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing par2 on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "par2"], check=False, timeout=300
            )

        elif os_type == "Windows":
            if _install_via_choco("par2cmdline"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Chocolatey failed, using WSL...")
                if _ensure_wsl():
                    _run_in_wsl("sudo apt update -y && sudo apt install -y par2")
                    success, output = _run_in_wsl("par2 -V")
                    if success:
                        print("✓ par2 installed in WSL")
                        return "par2 (WSL)"
                return "par2 (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "par2"], check=False, timeout=300)

        if shutil.which("par2"):
            success, output = _run_command(["par2", "-V"])
            if success:
                version_line = (
                    output.split("\n")[0] if output else "par2 (version unknown)"
                )
                print(f"✓ par2 installed: {version_line}")
                return version_line

        if os_type == "Windows":
            success, output = _run_in_wsl("par2 -V")
            if success:
                print("✓ par2 available in WSL")
                return "par2 (WSL)"

        return "par2 (unavailable)"

    except Exception as e:
        print(f"⚠ par2 error: {e}")
        return "par2 (unavailable)"


def ensure_ollama(min_version: str = "0.1") -> str:
    """Ensure Ollama is installed with WSL fallback."""
    if shutil.which("ollama"):
        success, output = _run_command(["ollama", "--version"])
        if success:
            version_line = output.strip()
            print(f"✓ Ollama already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing Ollama on {os_type}...")

    try:
        if os_type == "Linux":
            install_script = "curl -fsSL https://ollama.com/install.sh | sh"
            subprocess.run(install_script, shell=True, check=False, timeout=300)

        elif os_type == "Windows":
            if _install_via_winget("Ollama.Ollama"):
                print("✓ Installed via winget")
            else:
                print("⚠ Winget failed, using WSL...")
                if _ensure_wsl():
                    _run_in_wsl("curl -fsSL https://ollama.com/install.sh | sh")
                    success, output = _run_in_wsl("ollama --version")
                    if success:
                        print("✓ Ollama installed in WSL")
                        return "ollama (WSL)"
                return "ollama (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ollama"], check=False, timeout=300)

        if shutil.which("ollama"):
            success, output = _run_command(["ollama", "--version"])
            if success:
                version_line = output.strip()
                print(f"✓ Ollama installed: {version_line}")
                return version_line

        if os_type == "Windows":
            success, output = _run_in_wsl("ollama --version")
            if success:
                print("✓ Ollama available in WSL")
                return "ollama (WSL)"

        return "ollama (unavailable)"

    except Exception as e:
        print(f"⚠ Ollama error: {e}")
        return "ollama (unavailable)"


def ensure_java(min_version: int = 11) -> str:
    """Ensure Java is installed with WSL fallback."""
    if shutil.which("java"):
        success, output = _run_command(["java", "-version"])
        if success:
            import re

            version_output = output.lower()
            match = re.search(r'version "?(\d+)\.?(\d+)?', version_output)
            if match:
                major = int(match.group(1))
                if major == 1 and match.group(2):
                    major = int(match.group(2))

                version_line = output.split("\n")[0]
                if major >= min_version:
                    print(f"✓ Java already installed: {version_line}")
                    return version_line

    os_type = _get_os()
    print(f"Installing Java {min_version}+ on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", f"openjdk-{min_version}-jdk"],
                check=False,
                timeout=300,
            )

        elif os_type == "Windows":
            if _install_via_winget("EclipseAdoptium.Temurin.11.JDK"):
                print("✓ Installed via winget")
            elif _install_via_choco(f"openjdk{min_version}"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Native install failed, using WSL...")
                if _ensure_wsl():
                    _run_in_wsl(
                        f"sudo apt update -y && sudo apt install -y openjdk-{min_version}-jdk"
                    )
                    success, output = _run_in_wsl("java -version")
                    if success:
                        print("✓ Java installed in WSL")
                        return f"java-{min_version} (WSL)"
                return "java (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", f"openjdk@{min_version}"],
                    check=False,
                    timeout=300,
                )

        if shutil.which("java"):
            success, output = _run_command(["java", "-version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ Java installed: {version_line}")
                return version_line

        if os_type == "Windows":
            success, output = _run_in_wsl("java -version")
            if success:
                print("✓ Java available in WSL")
                return "java (WSL)"

        return "java (unavailable)"

    except Exception as e:
        print(f"⚠ Java error: {e}")
        return "java (unavailable)"


# Keep other functions (libpff_python, apache_tika, pdfarranger) as they were


def ensure_libpff_python(min_version: str = "20180714") -> str:
    """
    Ensure libpff-python is installed for reading Outlook PST/OST files.

    Args:
                    min_version: Minimum required version (default: "20180714")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If libpff-python cannot be installed or verified

    Example verification commands:
                    python -c "import pypff; print(pypff.__version__)"
    """
    # Check if already installed
    try:
        import pypff

        version = getattr(pypff, "__version__", "unknown")
        print(f"✓ libpff-python already installed: version {version}")
        return f"libpff-python {version}"
    except ImportError:
        pass

    # Install based on OS
    os_type = _get_os()
    print(f"Installing libpff-python on {os_type}...")

    try:
        if os_type == "Linux":
            # Install system dependencies first
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "libpff-dev"], check=True, timeout=300
            )
            # Install Python package
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "libpff-python"],
                check=True,
                timeout=300,
            )
        elif os_type == "Windows":
            # Windows typically uses pre-built wheels
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "libpff-python"],
                check=True,
                timeout=300,
            )
        elif os_type == "Darwin":
            # Install dependencies via Homebrew
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "libpff"], check=True, timeout=300)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "libpff-python"],
                check=True,
                timeout=300,
            )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        try:
            import pypff

            version = getattr(pypff, "__version__", "unknown")
            print(f"✓ libpff-python installed successfully: version {version}")
            print(
                f"  Verify with: {sys.executable} -c 'import pypff; print(pypff.__version__)'"
            )
            return f"libpff-python {version}"
        except ImportError:
            raise ModuleNotFoundError(
                "libpff-python pip install succeeded but module cannot be imported"
            )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"libpff-python installation failed: {e}")


def ensure_java(min_version: int = 11) -> str:
    """
    Ensure Java (JDK/JRE) is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required major version (default: 11)

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If Java cannot be installed or version is insufficient

    Example verification commands:
                    java -version
                    javac -version
    """
    # Check if already installed
    if shutil.which("java"):
        success, output = _run_command(["java", "-version"])
        if success:
            # Java version output goes to stderr typically
            version_output = output.lower()
            # Parse version (handles both old and new format)
            import re

            match = re.search(r'version "?(\d+)\.?(\d+)?', version_output)
            if match:
                major = int(match.group(1))
                # Java 9+ uses single digit versioning (11, 17, etc)
                # Java 8 and below use 1.x format
                if major == 1 and match.group(2):
                    major = int(match.group(2))

                version_line = output.split("\n")[0]
                if major >= min_version:
                    print(f"✓ Java already installed: {version_line}")
                    return version_line
                else:
                    print(f"⚠ Java {major} found, but need version {min_version}+")

    # Install based on OS
    os_type = _get_os()
    print(f"Installing Java {min_version}+ on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", f"openjdk-{min_version}-jdk"],
                check=True,
                timeout=300,
            )
        elif os_type == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    [
                        "winget",
                        "install",
                        "--id",
                        "EclipseAdoptium.Temurin.11.JDK",
                        "-e",
                    ],
                    check=True,
                    timeout=300,
                )
            elif shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", f"openjdk{min_version}"],
                    check=True,
                    timeout=300,
                )
            else:
                raise ModuleNotFoundError(
                    f"Java installation failed: No package manager found.\n"
                    f"Download from: https://adoptium.net/"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", f"openjdk@{min_version}"],
                    check=True,
                    timeout=300,
                )
            else:
                raise ModuleNotFoundError(
                    "Java installation failed: Homebrew not found.\n"
                    f"Install with: brew install openjdk@{min_version}"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("java"):
            success, output = _run_command(["java", "-version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ Java installed successfully: {version_line}")
                print("  Verify with: java -version")
                return version_line

        raise ModuleNotFoundError(
            "Java installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"Java installation failed: {e}")


def ensure_apache_tika(tika_version: str = "2.9.2", install_dir: str = None) -> str:
    """
    Ensure Apache Tika JAR is downloaded and available.

    Args:
                    tika_version: Version to download (default: "2.9.2")
                    install_dir: Directory to install Tika JAR (default: ~/.tika/)

    Returns:
                    Path to the Tika JAR file

    Raises:
                    ModuleNotFoundError: If Tika cannot be downloaded or Java is not available

    Example verification commands:
                    java -jar /path/to/tika-app.jar --version
                    java -jar /path/to/tika-app.jar --help
    """
    import os
    import urllib.request

    # Ensure Java is installed first
    try:
        ensure_java(min_version=11)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Apache Tika requires Java 11+: {e}")

    # Set install directory
    if install_dir is None:
        install_dir = os.path.expanduser("~/.tika")

    os.makedirs(install_dir, exist_ok=True)

    jar_name = f"tika-app-{tika_version}.jar"
    jar_path = os.path.join(install_dir, jar_name)

    # Check if already downloaded
    if os.path.exists(jar_path):
        # Verify it works
        success, output = _run_command(["java", "-jar", jar_path, "--version"])
        if success and tika_version in output:
            print(f"✓ Apache Tika already installed: {jar_path}")
            print(f"  Version: {tika_version}")
            return jar_path

    # Download Tika
    print(f"Downloading Apache Tika {tika_version}...")
    tika_url = f"https://archive.apache.org/dist/tika/{tika_version}/{jar_name}"

    try:
        print(f"  From: {tika_url}")
        print(f"  To: {jar_path}")

        urllib.request.urlretrieve(tika_url, jar_path)

        # Verify download
        if (
            os.path.exists(jar_path) and os.path.getsize(jar_path) > 1000000
        ):  # Should be >1MB
            success, output = _run_command(["java", "-jar", jar_path, "--version"])
            if success:
                print(f"✓ Apache Tika downloaded successfully: {jar_path}")
                print(f"  Verify with: java -jar {jar_path} --version")
                return jar_path

        raise ModuleNotFoundError("Tika JAR downloaded but verification failed")

    except Exception as e:
        # Clean up partial download
        if os.path.exists(jar_path):
            os.remove(jar_path)
        raise ModuleNotFoundError(f"Apache Tika download failed: {e}")


def ensure_pdfarranger(min_version: str = "1.8") -> str:
    """
    Ensure pdfarranger is installed for PDF manipulation.

    Args:
                    min_version: Minimum required version (default: "1.8")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If pdfarranger cannot be installed or verified

    Example verification commands:
                    pdfarranger --version
                    python -c "import pdfarranger"
    """
    # Check if already installed
    if shutil.which("pdfarranger"):
        success, output = _run_command(["pdfarranger", "--version"])
        if success:
            version_line = output.strip()
            print(f"✓ pdfarranger already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing pdfarranger on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "pdfarranger"], check=True, timeout=300
            )
        elif os_type == "Windows":
            # Windows typically uses pip installation
            print("Installing pdfarranger via pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pdfarranger"],
                check=True,
                timeout=300,
            )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", "pdfarranger"], check=True, timeout=300
                )
            else:
                # Fallback to pip
                print("Homebrew not found, installing via pip...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "pdfarranger"],
                    check=True,
                    timeout=300,
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("pdfarranger"):
            success, output = _run_command(["pdfarranger", "--version"])
            if success:
                version_line = output.strip()
                print(f"✓ pdfarranger installed successfully: {version_line}")
                print("  Verify with: pdfarranger --version")
                return version_line

        raise ModuleNotFoundError(
            "pdfarranger installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"pdfarranger installation failed: {e}")


def ensure_ollama(min_version: str = "0.1") -> str:
    """
    Ensure Ollama is installed for running local LLMs.

    Args:
                    min_version: Minimum required version (default: "0.1")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If Ollama cannot be installed or verified

    Example verification commands:
                    ollama --version
                    ollama list
                    ollama serve (to start the server)
    """
    # Check if already installed
    if shutil.which("ollama"):
        success, output = _run_command(["ollama", "--version"])
        if success:
            version_line = output.strip()
            print(f"✓ Ollama already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing Ollama on {os_type}...")

    try:
        if os_type == "Linux":
            # Use official install script
            print("Running official Ollama install script...")
            install_script = "curl -fsSL https://ollama.com/install.sh | sh"
            subprocess.run(install_script, shell=True, check=True, timeout=300)
        elif os_type == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    ["winget", "install", "--id", "Ollama.Ollama", "-e"],
                    check=True,
                    timeout=300,
                )
            else:
                raise ModuleNotFoundError(
                    "Ollama installation failed: winget not found.\n"
                    "Download installer from: https://ollama.com/download/windows"
                )
        elif os_type == "Darwin":
            # Check if Homebrew available
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ollama"], check=True, timeout=300)
            else:
                # Suggest manual download
                raise ModuleNotFoundError(
                    "Ollama installation failed: Homebrew not found.\n"
                    "Download from: https://ollama.com/download/mac\n"
                    "Or install with: brew install ollama"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("ollama"):
            success, output = _run_command(["ollama", "--version"])
            if success:
                version_line = output.strip()
                print(f"✓ Ollama installed successfully: {version_line}")
                print("  Verify with: ollama --version")
                print("  Start server with: ollama serve")
                return version_line

        raise ModuleNotFoundError(
            "Ollama installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"Ollama installation failed: {e}")


def ensure_unpaper(min_version: str = "6.1") -> str:
    """
    Ensure unpaper is installed for cleaning up scanned images.

    Args:
        min_version: Minimum required version (default: "6.1")

    Returns:
        Installed version string

    Raises:
        ModuleNotFoundError: If unpaper cannot be installed or verified

    Example verification commands:
        unpaper --version
        unpaper --help
    """
    # Check if already installed
    if shutil.which("unpaper"):
        success, output = _run_command(["unpaper", "--version"])
        if success:
            version_line = output.strip().split("\n")[0]
            print(f"✓ unpaper already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing unpaper on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "unpaper"], check=True, timeout=300
            )
        elif os_type == "Windows":
            if shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", "unpaper"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "unpaper installation failed: Chocolatey not found.\n"
                    "Install Chocolatey from: https://chocolatey.org/install\n"
                    "Then run: choco install unpaper\n"
                    "Or download binaries from: https://github.com/unpaper/unpaper/releases"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "unpaper"], check=True, timeout=300)
            else:
                raise ModuleNotFoundError(
                    "unpaper installation failed: Homebrew not found.\n"
                    "Install Homebrew from: https://brew.sh\n"
                    "Then run: brew install unpaper"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("unpaper"):
            success, output = _run_command(["unpaper", "--version"])
            if success:
                version_line = output.strip().split("\n")[0]
                print(f"✓ unpaper installed successfully: {version_line}")
                print("  Verify with: unpaper --version")
                return version_line

        raise ModuleNotFoundError(
            "unpaper installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"unpaper installation failed: {e}")


def ensure_pngquant(min_version: str = "2.0") -> str:
    """Ensure pngquant is installed with WSL fallback."""
    # Check if already installed
    if shutil.which("pngquant"):
        success, output = _run_command(["pngquant", "--version"])
        if success:
            version_line = output.strip().split("\n")[0]
            print(f"✓ pngquant already installed: {version_line}")
            return version_line

    os_type = _get_os()
    print(f"Installing pngquant on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update", "-y"], check=False, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "pngquant"], check=False, timeout=300
            )

        elif os_type == "Windows":
            # Try chocolatey first
            if _install_via_choco("pngquant"):
                print("✓ Installed via chocolatey")
            else:
                print("⚠ Chocolatey failed, using WSL...")
                if _ensure_wsl():
                    print("Installing pngquant in WSL...")
                    _run_in_wsl("sudo apt update -y && sudo apt install -y pngquant")
                    success, output = _run_in_wsl("pngquant --version")
                    if success:
                        print("✓ pngquant installed in WSL")
                        return "pngquant (WSL)"
                    else:
                        print("⚠ WSL install incomplete")
                else:
                    print("⚠ WSL unavailable. pngquant will not be available.")
                    print("   PDF optimization will be disabled.")
                return "pngquant (unavailable)"

        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", "pngquant"], check=False, timeout=300
                )

        # Verify installation
        if shutil.which("pngquant"):
            success, output = _run_command(["pngquant", "--version"])
            if success:
                version_line = output.strip().split("\n")[0]
                print(f"✓ pngquant installed: {version_line}")
                return version_line

        # Check WSL as fallback
        if os_type == "Windows":
            success, output = _run_in_wsl("pngquant --version")
            if success:
                print("✓ pngquant available in WSL")
                return "pngquant (WSL)"

        print("⚠ pngquant not installed, optimization will be disabled")
        return "pngquant (unavailable)"

    except Exception as e:
        print(f"⚠ pngquant installation issue: {e}")
        print("   PDF optimization will be limited")
        return "pngquant (unavailable)"


def _refresh_windows_path():
    """Refresh PATH on Windows after installation."""
    if _get_os() != "Windows":
        return

    try:
        import winreg
        import os

        # Read system PATH
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        )
        system_path = winreg.QueryValueEx(key, "PATH")[0]
        winreg.CloseKey(key)

        # Read user PATH
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        user_path = winreg.QueryValueEx(key, "PATH")[0]
        winreg.CloseKey(key)

        # Update current process PATH
        os.environ["PATH"] = system_path + ";" + user_path + ";" + os.environ["PATH"]
    except:
        pass


def ensure_tesseract(min_version: str = "4.0") -> str:
    """Ensure Tesseract OCR is installed."""
    if shutil.which("tesseract"):
        success, output = _run_command(["tesseract", "--version"])
        if success:
            print(f"✓ Tesseract already installed: {output.strip().split(chr(10))[0]}")
            return output.strip().split("\n")[0]

    os_type = _get_os()
    print(f"Installing Tesseract OCR on {os_type}...")

    if os_type == "Windows":
        # Try chocolatey
        if shutil.which("choco"):
            result = subprocess.run(
                ["choco", "install", "-y", "tesseract"],
                capture_output=True,
                timeout=300,
                check=False,
            )
            if result.returncode == 0:
                print("✓ Chocolatey install completed, refreshing PATH...")
                _refresh_windows_path()  # Refresh PATH

                # Check again
                if shutil.which("tesseract"):
                    print("✓ Tesseract verified in PATH")
                    return "tesseract (installed)"
                else:
                    print("⚠ Tesseract installed but not in PATH yet")
                    print("   Close and reopen your terminal, then run again")
                    return "tesseract (restart required)"

    return "tesseract (unavailable)"


def ensure_ghostscript(min_version: str = "9.50") -> str:
    """Ensure Ghostscript is installed."""
    gs_cmd = "gswin64c" if _get_os() == "Windows" else "gs"

    if shutil.which(gs_cmd) or shutil.which("gs"):
        cmd = gs_cmd if shutil.which(gs_cmd) else "gs"
        success, output = _run_command([cmd, "--version"])
        if success:
            print(f"✓ Ghostscript already installed: {output.strip()}")
            return output.strip()

    os_type = _get_os()
    print(f"Installing Ghostscript on {os_type}...")

    if os_type == "Windows":
        if shutil.which("choco"):
            result = subprocess.run(
                ["choco", "install", "-y", "ghostscript"],
                capture_output=True,
                timeout=300,
                check=False,
            )
            if result.returncode == 0:
                print("✓ Chocolatey install completed, refreshing PATH...")
                _refresh_windows_path()

                if shutil.which("gswin64c") or shutil.which("gs"):
                    print("✓ Ghostscript verified in PATH")
                    return "ghostscript (installed)"
                else:
                    print("⚠ Ghostscript installed but not in PATH yet")
                    print("   Close and reopen your terminal, then run again")
                    return "ghostscript (restart required)"

    return "ghostscript (unavailable)"


def ensure_jbig2enc() -> str:
    """Ensure jbig2enc is installed for better PDF compression."""
    if shutil.which("jbig2"):
        print("✓ jbig2enc already installed")
        return "jbig2enc (installed)"

    os_type = _get_os()
    print(f"Installing jbig2enc on {os_type}...")

    if os_type == "Windows":
        if shutil.which("choco"):
            subprocess.run(
                ["choco", "install", "-y", "jbig2enc"], check=False, timeout=300
            )
            _refresh_windows_path()
            if shutil.which("jbig2"):
                print("✓ jbig2enc installed")
                return "jbig2enc (installed)"

    print("⚠ jbig2enc unavailable (optional, PDFs will be larger)")
    return "jbig2enc (unavailable)"


def ensure_cuda_gpu(min_cuda_version: str = "11.0") -> dict:
    """
    Ensure CUDA-capable GPU with proper drivers is available.

    Args:
                    min_cuda_version: Minimum required CUDA version (default: "11.0")

    Returns:
                    Dict containing:
                                    - status: str ("available", "partial", "unavailable")
                                    - cuda_version: str or None
                                    - driver_version: str or None
                                    - gpu_name: str or None
                                    - gpu_memory_gb: float or None
                                    - compute_capability: str or None
                                    - message: str with detailed information

    Example verification commands:
                    nvidia-smi
                    nvcc --version
                    python -c "import torch; print(torch.cuda.is_available())"
    """
    import re

    result = {
        "status": "unavailable",
        "cuda_version": None,
        "driver_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "compute_capability": None,
        "message": "",
    }

    print("=" * 60)
    print("CUDA & GPU Detection")
    print("=" * 60)

    # Check 1: nvidia-smi (driver and GPU detection)
    print("\n[1/4] Checking NVIDIA drivers and GPU...")
    if shutil.which("nvidia-smi"):
        success, output = _run_command(["nvidia-smi"], timeout=10)
        if success:
            print("✓ nvidia-smi found and working")

            # Parse driver version
            driver_match = re.search(r"Driver Version: ([\d.]+)", output)
            if driver_match:
                result["driver_version"] = driver_match.group(1)
                print(f"  Driver Version: {result['driver_version']}")

            # Parse CUDA version from driver
            cuda_match = re.search(r"CUDA Version: ([\d.]+)", output)
            if cuda_match:
                cuda_from_driver = cuda_match.group(1)
                print(f"  CUDA Version (from driver): {cuda_from_driver}")

            # Parse GPU name
            gpu_match = re.search(
                r"(?:NVIDIA|GeForce|Tesla|Quadro|RTX)\s+([^\n|]+)", output
            )
            if gpu_match:
                result["gpu_name"] = gpu_match.group(0).strip()
                print(f"  GPU: {result['gpu_name']}")

            # Parse GPU memory
            mem_match = re.search(r"(\d+)MiB\s+/\s+(\d+)MiB", output)
            if mem_match:
                total_mem_mb = int(mem_match.group(2))
                result["gpu_memory_gb"] = total_mem_mb / 1024
                print(f"  GPU Memory: {result['gpu_memory_gb']:.1f} GB")

            result["status"] = "partial"
        else:
            print("⚠ nvidia-smi found but failed to execute")
            print("  This usually means driver issues")
    else:
        print("✗ nvidia-smi not found")
        print("  NVIDIA drivers are not installed or not in PATH")

    # Check 2: NVCC (CUDA toolkit)
    print("\n[2/4] Checking CUDA toolkit (nvcc)...")
    if shutil.which("nvcc"):
        success, output = _run_command(["nvcc", "--version"], timeout=10)
        if success:
            version_match = re.search(r"release ([\d.]+)", output)
            if version_match:
                result["cuda_version"] = version_match.group(1)
                print(f"✓ CUDA toolkit installed: {result['cuda_version']}")

                # Check if meets minimum version
                try:
                    cuda_major = float(result["cuda_version"].split(".")[0])
                    min_major = float(min_cuda_version.split(".")[0])

                    if cuda_major >= min_major:
                        print(f"  Meets minimum requirement ({min_cuda_version})")
                    else:
                        print(f"⚠ Below minimum requirement ({min_cuda_version})")
                        print(f"  Consider upgrading CUDA toolkit")
                except:
                    pass
            else:
                print("✓ nvcc found but couldn't parse version")
        else:
            print("⚠ nvcc found but failed to execute")
    else:
        print("✗ nvcc not found")
        print("  CUDA toolkit is not installed or not in PATH")

    # Check 3: PyTorch CUDA support
    print("\n[3/4] Checking PyTorch CUDA support...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA available: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA devices found: {torch.cuda.device_count()}")

            if torch.cuda.device_count() > 0:
                device_name = torch.cuda.get_device_name(0)
                print(f"  Primary GPU: {device_name}")

                # Get compute capability
                props = torch.cuda.get_device_properties(0)
                compute_cap = f"{props.major}.{props.minor}"
                result["compute_capability"] = compute_cap
                print(f"  Compute Capability: {compute_cap}")

                # Update GPU info if not found earlier
                if not result["gpu_name"]:
                    result["gpu_name"] = device_name
                if not result["gpu_memory_gb"]:
                    result["gpu_memory_gb"] = props.total_memory / (1024**3)
                    print(f"  GPU Memory: {result['gpu_memory_gb']:.1f} GB")

                result["status"] = "available"
        else:
            print("✗ PyTorch found but CUDA not available")
            print("  You may have CPU-only PyTorch installed")
            print(
                "  Reinstall with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
    except ImportError:
        print("⚠ PyTorch not installed")
        print(
            "  Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )
    except Exception as e:
        print(f"⚠ Error checking PyTorch: {e}")

    # Check 4: cuDNN (optional but recommended)
    print("\n[4/4] Checking cuDNN...")
    try:
        import torch

        if torch.cuda.is_available():
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
                print(f"✓ cuDNN available: {cudnn_version}")
            else:
                print("⚠ cuDNN not available")
                print("  Some operations may be slower")
    except:
        print("⚠ Could not check cuDNN status")

    # Generate summary message
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if result["status"] == "available":
        result["message"] = (
            f"✓ CUDA GPU fully available: {result['gpu_name']} ({result['gpu_memory_gb']:.1f} GB)"
        )
        print(f"✓ Status: FULLY AVAILABLE")
        print(f"  GPU: {result['gpu_name']}")
        print(f"  Memory: {result['gpu_memory_gb']:.1f} GB")
        print(f"  Driver: {result['driver_version']}")
        print(f"  CUDA: {result['cuda_version'] or 'N/A (using driver version)'}")
        print(f"  Compute Capability: {result['compute_capability']}")

    elif result["status"] == "partial":
        result["message"] = "⚠ GPU detected but CUDA support incomplete"
        print(f"⚠ Status: PARTIAL")
        print(f"  GPU detected: {result['gpu_name'] or 'Unknown'}")
        print(f"  Driver: {result['driver_version'] or 'Unknown'}")

        # Provide guidance
        print("\n  To enable full CUDA support:")
        if not result["cuda_version"]:
            print(
                "  1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
            )

        try:
            import torch

            if not torch.cuda.is_available():
                print("  2. Reinstall PyTorch with CUDA:")
                print(
                    "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                )
        except ImportError:
            print("  2. Install PyTorch with CUDA:")
            print(
                "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

    else:
        result["message"] = "✗ CUDA GPU not available"
        print(f"✗ Status: NOT AVAILABLE")

        os_type = _get_os()
        print(f"\n  System: {os_type}")
        print("  To enable CUDA GPU support:")

        if os_type == "Windows":
            print("\n  1. Install NVIDIA drivers:")
            print("     - Visit: https://www.nvidia.com/download/index.aspx")
            print("     - Or use GeForce Experience for automatic updates")
            print("\n  2. Install CUDA Toolkit:")
            print("     - Download from: https://developer.nvidia.com/cuda-downloads")
            print("     - Recommended version: 11.8 or 12.1")
            print("\n  3. Install PyTorch with CUDA:")
            print(
                "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        elif os_type == "Linux":
            print("\n  1. Install NVIDIA drivers:")
            print("     sudo apt update")
            print("     sudo apt install nvidia-driver-535")  # Recent stable version
            print("\n  2. Install CUDA Toolkit:")
            print(
                "     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
            )
            print("     sudo dpkg -i cuda-keyring_1.0-1_all.deb")
            print("     sudo apt-get update")
            print("     sudo apt-get install cuda")
            print("\n  3. Install PyTorch with CUDA:")
            print(
                "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        elif os_type == "Darwin":
            print("\n  ⚠ CUDA is not supported on macOS")
            print(
                "  Apple Silicon Macs can use Metal Performance Shaders (MPS) with PyTorch"
            )
            print("  For MPS support:")
            print("     pip install torch torchvision")
            print("  Then use: torch.device('mps') instead of torch.device('cuda')")

    print("=" * 60)

    return result


# Helper function to get detailed CUDA info
def get_cuda_info() -> dict:
    """
    Get detailed CUDA and GPU information.

    Returns:
                    Dict with comprehensive CUDA/GPU details
    """
    info = ensure_cuda_gpu()

    # Add additional PyTorch-specific info if available
    try:
        import torch

        if torch.cuda.is_available():
            info["pytorch_cuda_version"] = torch.version.cuda
            info["pytorch_version"] = torch.__version__
            info["cudnn_version"] = (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            )
            info["device_count"] = torch.cuda.device_count()

            # Per-device info
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory_gb": props.total_memory / (1024**3),
                        "multi_processor_count": props.multi_processor_count,
                    }
                )
            info["devices"] = devices
    except:
        pass

    return info

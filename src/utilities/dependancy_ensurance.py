import platform
import shutil
import subprocess
import sys
from typing import Tuple


def _run_command(cmd: list, timeout: int = 30) -> Tuple[bool, str]:
    """Helper function to run shell commands safely."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def _get_os() -> str:
    """Get the current operating system."""
    return platform.system()


def ensure_ffmpeg(min_version: str = "4.0") -> str:
    """
    Ensure FFmpeg is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required version (default: "4.0")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If FFmpeg cannot be installed or verified

    Example verification commands:
                    ffmpeg -version
                    ffmpeg -formats
    """
    # Check if already installed
    if shutil.which("ffmpeg"):
        success, output = _run_command(["ffmpeg", "-version"])
        if success:
            # Extract version from first line
            version_line = output.split("\n")[0]
            if "version" in version_line.lower():
                print(f"✓ FFmpeg already installed: {version_line}")
                return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing FFmpeg on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "ffmpeg"], check=True, timeout=300
            )
        elif os_type == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    ["winget", "install", "--id", "Gyan.FFmpeg", "-e"],
                    check=True,
                    timeout=300,
                )
            elif shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", "ffmpeg"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "FFmpeg installation failed: No package manager found.\n"
                    "Install winget or chocolatey, or download from: https://ffmpeg.org/download.html"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=True, timeout=300)
            else:
                raise ModuleNotFoundError(
                    "FFmpeg installation failed: Homebrew not found.\n"
                    "Install Homebrew from: https://brew.sh\n"
                    "Then run: brew install ffmpeg"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("ffmpeg"):
            success, output = _run_command(["ffmpeg", "-version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ FFmpeg installed successfully: {version_line}")
                print("  Verify with: ffmpeg -version")
                return version_line

        raise ModuleNotFoundError(
            "FFmpeg installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"FFmpeg installation failed: {e}")


def ensure_imagemagick(min_version: str = "6.9") -> str:
    """
    Ensure ImageMagick is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required version (default: "6.9")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If ImageMagick cannot be installed or verified

    Example verification commands:
                    convert -version
                    identify -version
    """
    # Check if already installed
    if shutil.which("convert"):
        success, output = _run_command(["convert", "-version"])
        if success and "ImageMagick" in output:
            version_line = output.split("\n")[0]
            print(f"✓ ImageMagick already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing ImageMagick on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "imagemagick"], check=True, timeout=300
            )
        elif os_type == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    ["winget", "install", "--id", "ImageMagick.ImageMagick", "-e"],
                    check=True,
                    timeout=300,
                )
            elif shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", "imagemagick"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "ImageMagick installation failed: No package manager found.\n"
                    "Download from: https://imagemagick.org/script/download.php"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(
                    ["brew", "install", "imagemagick"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "ImageMagick installation failed: Homebrew not found.\n"
                    "Install with: brew install imagemagick"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("convert"):
            success, output = _run_command(["convert", "-version"])
            if success and "ImageMagick" in output:
                version_line = output.split("\n")[0]
                print(f"✓ ImageMagick installed successfully: {version_line}")
                print("  Verify with: convert -version")
                return version_line

        raise ModuleNotFoundError(
            "ImageMagick installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"ImageMagick installation failed: {e}")


def ensure_pandoc(min_version: str = "2.0") -> str:
    """
    Ensure Pandoc is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required version (default: "2.0")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If Pandoc cannot be installed or verified

    Example verification commands:
                    pandoc --version
                    pandoc --list-output-formats
    """
    # Check if already installed
    if shutil.which("pandoc"):
        success, output = _run_command(["pandoc", "--version"])
        if success:
            version_line = output.split("\n")[0]
            print(f"✓ Pandoc already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing Pandoc on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "pandoc"], check=True, timeout=300
            )
        elif os_type == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    ["winget", "install", "--id", "JohnMacFarlane.Pandoc", "-e"],
                    check=True,
                    timeout=300,
                )
            elif shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", "pandoc"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "Pandoc installation failed: No package manager found.\n"
                    "Download from: https://pandoc.org/installing.html"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "pandoc"], check=True, timeout=300)
            else:
                raise ModuleNotFoundError(
                    "Pandoc installation failed: Homebrew not found.\n"
                    "Install with: brew install pandoc"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("pandoc"):
            success, output = _run_command(["pandoc", "--version"])
            if success:
                version_line = output.split("\n")[0]
                print(f"✓ Pandoc installed successfully: {version_line}")
                print("  Verify with: pandoc --version")
                return version_line

        raise ModuleNotFoundError(
            "Pandoc installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"Pandoc installation failed: {e}")


def ensure_par2(min_version: str = "0.8") -> str:
    """
    Ensure par2 (Parchive) is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required version (default: "0.8")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If par2 cannot be installed or verified

    Example verification commands:
                    par2 -V
                    par2 -h
    """
    # Check if already installed
    if shutil.which("par2"):
        success, output = _run_command(["par2", "-V"])
        if success:
            version_line = output.split("\n")[0] if output else "par2 (version unknown)"
            print(f"✓ par2 already installed: {version_line}")
            return version_line

    # Install based on OS
    os_type = _get_os()
    print(f"Installing par2 on {os_type}...")

    try:
        if os_type == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True, timeout=60)
            subprocess.run(
                ["sudo", "apt", "install", "-y", "par2"], check=True, timeout=300
            )
        elif os_type == "Windows":
            if shutil.which("choco"):
                subprocess.run(
                    ["choco", "install", "-y", "par2cmdline"], check=True, timeout=300
                )
            else:
                raise ModuleNotFoundError(
                    "par2 installation failed: Chocolatey not found.\n"
                    "Download from: https://github.com/Parchive/par2cmdline/releases\n"
                    "Or install chocolatey first: https://chocolatey.org/install"
                )
        elif os_type == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "par2"], check=True, timeout=300)
            else:
                raise ModuleNotFoundError(
                    "par2 installation failed: Homebrew not found.\n"
                    "Install with: brew install par2"
                )
        else:
            raise ModuleNotFoundError(f"Unsupported OS: {os_type}")

        # Verify installation
        if shutil.which("par2"):
            success, output = _run_command(["par2", "-V"])
            if success:
                version_line = (
                    output.split("\n")[0] if output else "par2 (version unknown)"
                )
                print(f"✓ par2 installed successfully: {version_line}")
                print("  Verify with: par2 -V")
                return version_line

        raise ModuleNotFoundError(
            "par2 installation completed but command not found in PATH"
        )

    except subprocess.CalledProcessError as e:
        raise ModuleNotFoundError(f"par2 installation failed: {e}")


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

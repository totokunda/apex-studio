#!/bin/bash
# install.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get environment name from command line argument, default to 'apex'
ENV_NAME=${1:-apex}
print_status "Environment name: $ENV_NAME"

# Detect OS
OS="unknown"
DISTRO=""
DISTRO_VERSION=""

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
    else
        DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
        DISTRO_VERSION=$(lsb_release -rs)
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
fi
print_status "Detected OS: $OS, Distro: $DISTRO, Version: $DISTRO_VERSION"

# Get number of CPUs for job parallelism
if [[ "$OS" == "mac" ]]; then
    NUM_CPUS=$(sysctl -n hw.ncpu)
else
    # Default to nproc for Linux
    NUM_CPUS=$(nproc)
fi
MAX_JOBS=$((NUM_CPUS > 16 ? 16 : NUM_CPUS))


install_dependencies() {
    print_status "Checking and installing dependencies..."
    if [[ "$OS" == "linux" ]]; then
        # For Debian/Ubuntu
        if command -v apt-get &> /dev/null; then
            PACKAGES_TO_INSTALL=""
            if ! command -v git &> /dev/null; then PACKAGES_TO_INSTALL+="git "; fi
            if ! command -v wget &> /dev/null; then PACKAGES_TO_INSTALL+="wget "; fi
            if ! command -v lsb_release &> /dev/null; then PACKAGES_TO_INSTALL+="lsb-release "; fi
            if ! command -v gpg &> /dev/null; then PACKAGES_TO_INSTALL+="gpg "; fi
            
            if [[ -n "$PACKAGES_TO_INSTALL" ]]; then
                print_status "Installing: $PACKAGES_TO_INSTALL"
                sudo apt-get update
                sudo apt-get install -y $PACKAGES_TO_INSTALL
            fi
        # For RHEL/CentOS/Fedora
        elif command -v dnf &> /dev/null || command -v yum &> /dev/null; then
            INSTALL_CMD=""
            if command -v dnf &> /dev/null; then INSTALL_CMD="sudo dnf"; else INSTALL_CMD="sudo yum"; fi
            
            PACKAGES_TO_INSTALL=""
            if ! command -v git &> /dev/null; then PACKAGES_TO_INSTALL+="git "; fi
            if ! command -v wget &> /dev/null; then PACKAGES_TO_INSTALL+="wget "; fi
            if ! command -v lsb_release &> /dev/null; then PACKAGES_TO_INSTALL+="redhat-lsb-core "; fi
            if ! command -v gpg &> /dev/null; then PACKAGES_TO_INSTALL+="gnupg2 "; fi

            # For CUDA install on RHEL-based systems
            if command -v dnf &> /dev/null; then
                # dnf config-manager is in dnf-plugins-core
                if ! dnf info installed dnf-plugins-core &>/dev/null; then
                    PACKAGES_TO_INSTALL+="dnf-plugins-core ";
                fi
            elif command -v yum &> /dev/null; then
                # yum-config-manager is in yum-utils
                if ! command -v yum-config-manager &>/dev/null; then
                    PACKAGES_TO_INSTALL+="yum-utils ";
                fi
            fi

            if [[ -n "$PACKAGES_TO_INSTALL" ]]; then
                print_status "Installing: $PACKAGES_TO_INSTALL"
                $INSTALL_CMD install -y $PACKAGES_TO_INSTALL
            fi
        else
            print_warning "Could not find apt, dnf, or yum. Please install git, wget, lsb-release, and gpg manually."
        fi
    elif [[ "$OS" == "mac" ]]; then
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Please install Homebrew first from https://brew.sh/"
            exit 1
        fi
        if ! command -v git &> /dev/null; then brew install git; fi
        if ! command -v wget &> /dev/null; then brew install wget; fi
    fi
    print_success "Dependencies checked."
}

# Install dependencies at the beginning
install_dependencies

# Function to get CUDA version from nvidia-smi, supporting older drivers
get_cuda_version() {
    # nvidia-smi --query-gpu=cuda_version fails on older drivers.
    # We capture all output and check for the error message.
    local smi_output
    smi_output=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>&1 | head -n 1)

    if [[ "$smi_output" == *"not a valid field"* || "$smi_output" == *"not supported"* ]]; then
        # Fallback to parsing nvidia-smi output for older drivers
        smi_output=$(nvidia-smi 2>/dev/null)
        if [[ $? -eq 0 && -n "$smi_output" ]]; then
            local cuda_version
            cuda_version=$(echo "$smi_output" | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n 1)
            if [[ -n "$cuda_version" ]]; then
                echo "$cuda_version"
                return 0
            fi
        fi
    elif [[ -n "$smi_output" ]]; then
        # If the command succeeded, output is the version
        echo "$smi_output"
        return 0
    fi

    return 1
}

# Function to check if CUDA is available
check_cuda() {
    hash -r # Re-hash PATH to find newly installed commands
    if command -v nvidia-smi &> /dev/null; then
        # This command can fail if the driver is not loaded (e.g. requires reboot)
        DRIVER_CUDA_VERSION=$(get_cuda_version)
        if [[ $? -eq 0 && -n "$DRIVER_CUDA_VERSION" ]]; then
            print_success "NVIDIA driver detected, supporting CUDA $DRIVER_CUDA_VERSION"
            return 0
        fi
    fi
    return 1
}

install_cuda_debian() {
    print_status "Setting up CUDA 12 repository for Debian/Ubuntu..."
    local distro_id
    case "$DISTRO" in
        ubuntu)
            distro_id="ubuntu${DISTRO_VERSION//.}"
            ;;
        debian)
            distro_id="debian${DISTRO_VERSION}"
            ;;
    esac

    local cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro_id}/x86_64/"
    if ! wget -q --spider "$cuda_repo_url"; then
        print_error "Could not find CUDA repository for ${distro_id} at ${cuda_repo_url}."
        print_error "Your distribution may be too new or too old for the supported CUDA versions."
        return 1
    fi
    
    wget ${cuda_repo_url}cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    # Remove conflicting packages first
    print_status "Removing conflicting NVIDIA packages..."
    sudo apt-get remove -y nvidia-open || true
    sudo apt-get autoremove -y
    
    print_status "Installing CUDA 12 toolkit and NVIDIA drivers..."
    sudo apt-get -y install cuda-12-6
    return $?
}

install_cuda_rhel() {
    print_status "Setting up CUDA 12 repository for RHEL/CentOS/Fedora..."
    local distro_name="$DISTRO"
    if [[ "$DISTRO" == "centos" ]]; then
        distro_name="rhel"
    fi

    # Major version, e.g. 8 from 8.4
    local major_version="${DISTRO_VERSION%%.*}"
    
    # Check for Fedora separately as it has a different repo name structure
    if [[ "$DISTRO" == "fedora" ]]; then
        repo_url="https://developer.download.nvidia.com/compute/cuda/repos/fedora${major_version}/x86_64/cuda-fedora${major_version}.repo"
    else
        repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro_name}${major_version}/x86_64/cuda-${distro_name}${major_version}.repo"
    fi
    
    print_status "Adding CUDA repository..."
    if command -v dnf &> /dev/null; then
        sudo dnf config-manager --add-repo "$repo_url"
        sudo dnf clean all
        print_status "Installing CUDA 12 toolkit and NVIDIA drivers..."
        sudo dnf -y install cuda-12-6
    elif command -v yum &> /dev/null; then
        sudo yum-config-manager --add-repo "$repo_url"
        sudo yum clean all
        print_status "Installing CUDA 12 toolkit and NVIDIA drivers..."
        sudo yum -y install cuda-12-6
    else
        print_error "dnf or yum not found on RHEL-based system."
        return 1
    fi

    return $?
}

# Function to install CUDA on Linux
install_cuda() {
    print_status "Attempting to install CUDA and NVIDIA drivers..."
    
    local install_status
    case "$DISTRO" in
        ubuntu|debian)
            install_cuda_debian
            install_status=$?
            ;;
        centos|rhel|fedora)
            install_cuda_rhel
            install_status=$?
            ;;
        *)
            print_warning "Unsupported Linux distribution for automatic CUDA installation: $DISTRO"
            print_warning "Please install CUDA manually and re-run this script."
            return 1
            ;;
    esac

    if [ $install_status -ne 0 ]; then
        print_error "CUDA installation failed."
        return 1
    fi
    
    # Add CUDA to PATH. The installer often creates /usr/local/cuda symlink.
    if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
        print_status "Adding CUDA to PATH in ~/.bashrc"
        echo '' >> ~/.bashrc
        echo '# CUDA PATHS' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
        
        # Export for current shell
        export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA installation failed. nvcc not found in PATH."
        return 1
    fi

    local installed_toolkit_version
    installed_toolkit_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA Toolkit ${installed_toolkit_version} installed successfully"

    if [ -f /var/run/reboot-required ]; then
        print_warning "A reboot is required to complete the NVIDIA driver installation."
    fi

    return 0
}

# CUDA Setup
CUDA_AVAILABLE=false
CUDA_VERSION=""
if [[ "$OS" == "mac" ]]; then
    print_warning "macOS detected - skipping CUDA installation"
else
    # Check if CUDA toolkit is available
    if command -v nvcc &> /dev/null; then
        CUDA_AVAILABLE=true
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "CUDA toolkit detected: version $CUDA_VERSION"
    elif check_cuda; then
        CUDA_AVAILABLE=true
        CUDA_VERSION=$(get_cuda_version)
        print_success "NVIDIA driver detected but no CUDA toolkit, attempting to install..."
    else
        print_status "No CUDA toolkit or NVIDIA driver detected, attempting to install..."
    fi
    
    # Install CUDA if not available
    if [[ "$CUDA_AVAILABLE" == false ]] || [[ ! -z "$CUDA_VERSION" && "$CUDA_VERSION" != "12"* ]]; then
        print_status "Installing CUDA 12..."
        if install_cuda; then
            # Re-check after installation
            if command -v nvcc &> /dev/null; then
                CUDA_AVAILABLE=true
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                print_success "CUDA 12 installation successful: version $CUDA_VERSION"
            elif check_cuda; then
                CUDA_AVAILABLE=true
                CUDA_VERSION=$(get_cuda_version)
                print_success "CUDA 12 installation successful"
            else
                print_warning "Could not detect CUDA after installation. Some features may be limited."
                CUDA_AVAILABLE=false
            fi
        else
            print_warning "CUDA installation failed, continuing without CUDA support."
            CUDA_AVAILABLE=false
        fi
    fi
fi

# Function to install Miniconda
install_miniconda() {
    print_status "Installing Miniconda..."
    
    if [[ "$OS" == "mac" ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    
    wget $MINICONDA_URL -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    # To make conda available in the current script session
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
    print_success "Miniconda installed successfully"
}

# Check if Conda is available
if ! command -v conda &> /dev/null; then
    print_status "Conda not found, installing Miniconda..."
    install_miniconda
else
    print_success "Conda is already available"
    # Ensure conda is initialized for the current shell
    eval "$(conda shell.bash hook)"
fi

# Ensure conda is in PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Accept conda Terms of Service for default channels
print_status "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
if conda info --envs | grep -wq "^$ENV_NAME\s"; then
    print_status "Conda environment '$ENV_NAME' already exists."
else
    print_status "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
fi
# Using `conda run` is more reliable than `source activate` in scripts
CONDA_RUN="conda run -n $ENV_NAME"

# Install PyTorch
print_status "Installing PyTorch..."
if [[ "$CUDA_AVAILABLE" == true ]]; then
    print_status "Installing PyTorch with CUDA 12.6 support..."
    # For CUDA 12.6, use the cu126 wheel
    $CONDA_RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    
    # Clean up any conflicting MLX installations
    print_status "Cleaning up conflicting MLX installations..."
    $CONDA_RUN pip uninstall -y mlx mlx-cuda mlx-cpu || true
    
    # Install MLX with CPU support to avoid CUDA version conflicts
    print_status "Installing MLX with CPU support to avoid CUDA version conflicts..."
    $CONDA_RUN pip install 'mlx[cpu]'

    print_status "Verifying PyTorch CUDA setup..."
    if $CONDA_RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); exit(0) if torch.cuda.is_available() else exit(1)"; then
        print_success "PyTorch CUDA is available and working."
    else
        print_warning "PyTorch CUDA verification failed. This may be due to NVIDIA driver issues."
        print_warning "Continuing with installation - CUDA support may be limited until driver issues are resolved."
    fi
else
    if [[ "$OS" == "mac" ]]; then
        print_status "Installing PyTorch for macOS..."
        $CONDA_RUN pip install torch torchvision torchaudio mlx
    else
        print_status "Installing PyTorch with CPU support..."
        $CONDA_RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        $CONDA_RUN pip install 'mlx[cpu]'
    fi
fi

# Install requirements from requirements.txt
print_status "Installing requirements from requirements.txt..."
#$CONDA_RUN pip install -r requirements.txt

# Create thirdparty directory if it doesn't exist
mkdir -p thirdparty
cd thirdparty

git submodule update --init --recursive

# Function to clone and install a repository
clone_and_install() {
    local repo_url=$1
    local repo_name=$2
    local install_command=$3
    
    if [[ ! -d "$repo_name" ]]; then
        print_status "Cloning $repo_name..."
        git clone $repo_url
    else
        print_status "$repo_name directory already exists"
    fi
    
    cd $repo_name
    print_status "Installing $repo_name..."
    eval $install_command
    cd ..
}

# Install diffusers
print_status "Installing diffusers..."
if [[ ! -d "diffusers" ]]; then
    clone_and_install "https://github.com/huggingface/diffusers.git" "diffusers" "$CONDA_RUN pip install -e ."
else
    cd diffusers
    # git clone submodule
    $CONDA_RUN pip install -e .
    cd ..
fi

# check if GroundingDINO is installed in the conda environment and install it if not
if [[ ! -d "$CONDA_PREFIX/lib/python3.10/site-packages/groundingdino" ]]; then
    print_status "Installing patch for gdino..."
    if [[ ! -d "GroundingDINO" ]]; then
        git clone "https://github.com/IDEA-Research/GroundingDINO.git" "GroundingDINO"
    fi
    
    cp ../patches/gdino-fixed.cu GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
    # install gdino
    cd GroundingDINO
    $CONDA_RUN pip install -e .
    cd ..
else
    print_status "GroundingDINO is already installed in the conda environment."
fi

# Check if we can support Hopper GPUs (Compute Capability 9.0+)
SUPPORTS_HOPPER=false
if [[ "$CUDA_AVAILABLE" == true ]]; then
    if command -v nvidia-smi &> /dev/null; then
        # Get major compute capability, e.g., 8 from 8.6
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | cut -d'.' -f1)
        if [[ -n "$COMPUTE_CAP" && "$COMPUTE_CAP" -ge 9 ]]; then
            SUPPORTS_HOPPER=true
            print_success "Hopper GPU architecture (Compute Capability 9.0+) detected"
        elif [[ -n "$COMPUTE_CAP" ]]; then
            print_status "Non-Hopper GPU detected (Compute Capability: ${COMPUTE_CAP}.x). Skipping Hopper-specific installations."
        else
            print_warning "Could not determine GPU compute capability. Skipping Hopper-specific installations."
        fi
    fi
fi

export MAX_JOBS=$MAX_JOBS

# Install Flash Attention
print_status "Installing Flash Attention (FlashAttention 3 wheels)..."
if [[ "$CUDA_AVAILABLE" == true ]]; then
    # Prefer prebuilt FlashAttention 3 wheels (CUDA 12.8 + Torch 2.9.0) which work on both
    # Linux and Windows. (Repo: https://windreamer.github.io/flash-attention3-wheels/)
    if ! $CONDA_RUN python -c "import flash_attn_interface" >/dev/null 2>&1; then
        print_status "Installing FlashAttention 3 wheels (cu128 + torch2.9.0)..."
        $CONDA_RUN pip install flash_attn_3 \
            --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch290 \
            --extra-index-url https://download.pytorch.org/whl/cu128
    else
        print_status "FlashAttention 3 is already installed in the conda environment."
    fi
else
    print_warning "CUDA not available, skipping installation."
fi

# Install SageAttention
print_status "Installing SageAttention..."
if [[ "$CUDA_AVAILABLE" == true ]]; then
    $CONDA_RUN pip install --upgrade pip setuptools wheel
    export SETUPTOOLS_USE_DISTUTILS=stdlib
    unset PYTHONPATH
    # check if SageAttention is installed in the conda environment and install it if not
    if [[ ! -d "$CONDA_PREFIX/lib/python3.10/site-packages/sageattention" ]]; then
        if [[ ! -d "SageAttention" ]]; then
            clone_and_install "https://github.com/thu-ml/SageAttention.git" "SageAttention" "$CONDA_RUN pip install . --verbose --no-build-isolation"
        else
            cd SageAttention
            # Try to install with GPU support first, fallback to CPU if it fails
            if ! $CONDA_RUN pip install . --verbose --no-build-isolation; then
                print_warning "SageAttention GPU installation failed, trying CPU fallback..."
                # Set environment variables to force CPU build
                export CUDA_VISIBLE_DEVICES=""
                export FORCE_CUDA=0
                if ! $CONDA_RUN pip install . --verbose --no-build-isolation; then
                    print_warning "SageAttention installation failed completely, skipping..."
                fi
            fi
            cd ..
        fi
    else
        print_status "SageAttention is already installed in the conda environment."
    fi
else
    print_warning "CUDA not available, skipping installation."
fi

# Return to main directory
cd ..

# Apply patches 
print_status "Applying patches..."
bash scripts/apply_patches.sh --python-cmd "$CONDA_RUN python"
print_success "Patches applied successfully."

source ~/.bashrc

# Final success message
print_success "🎉 Environment setup completed successfully!"
print_success "Environment name: $ENV_NAME"
print_success "CUDA available: $CUDA_AVAILABLE (Version: ${CUDA_VERSION:-N/A})"
print_success "Hopper GPU support: $SUPPORTS_HOPPER"
print_status "To activate your environment, run: conda activate $ENV_NAME"

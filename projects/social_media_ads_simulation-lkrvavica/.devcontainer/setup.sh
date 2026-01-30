#!/bin/bash
# ABOUTME: Optimized setup script for scientific programming environment
# This script creates a robust, student-friendly development environment
# Author: Scientific Programming Team
# Version: 2.0 - Student Optimized

set -e

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Error handling function
handle_error() {
    log_error "Setup failed at line $1: $2"
    log_info "Don't worry! Your container is still functional. You can:"
    echo "   1. Continue with basic Python development"
    echo "   2. Install packages manually with: uv add package_name"
    echo "   3. Ask your instructor for help if needed"
    exit 1
}

# Set up error handling
trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

# Start setup
log_info "🚀 Setting up Scientific Programming Environment..."

# Create essential directories with better error handling
log_info "📁 Creating project directories..."
mkdir -p ~/data ~/notebooks ~/reports ~/figures ~/models ~/templates

# Try to create directories in /workspaces if possible
if [ -w "/workspaces" ]; then
    mkdir -p /workspaces/reports /workspaces/figures /workspaces/models 2>/dev/null || true
    log_success "Created directories in /workspaces/"
else
    log_warning "Cannot write to /workspaces, using home directory instead"
    log_info "Home directories created: ~/data, ~/notebooks, ~/reports, ~/figures, ~/models"
fi

# Upgrade pip and install UV package manager
log_info "📦 Upgrading pip and installing UV..."
python -m pip install --upgrade pip --quiet || log_warning "Pip upgrade failed, continuing..."

# Install UV with better error handling
log_info "⚡ Installing UV package manager..."
UV_AVAILABLE=false
if command -v uv >/dev/null 2>&1; then
    log_success "UV is already installed"
    UV_AVAILABLE=true
else
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
        log_success "UV installed successfully"
        UV_AVAILABLE=true
    else
        log_error "Failed to install UV"
        log_info "You can still use pip for package management"
        UV_AVAILABLE=false
    fi
fi

# Skip project configuration - use direct package installation
log_info "⚙️ Skipping project configuration - using direct package installation..."

# Install packages with error handling
install_packages() {
    local packages=("$@")
    log_info "🔬 Installing packages: ${packages[*]}"

    # Check if we have a virtual environment available
    if [ -d ".venv" ] && [ "$UV_AVAILABLE" = true ]; then
        # Use UV with virtual environment
        if uv pip install "${packages[@]}" --quiet; then
            log_success "Installed: ${packages[*]}"
        else
            log_warning "UV installation failed for some packages, trying pip..."
            python -m pip install "${packages[@]}" --quiet || log_warning "Some packages failed to install"
        fi
    elif [ "$UV_AVAILABLE" = true ]; then
        # Use UV with system-wide installation
        if uv pip install "${packages[@]}" --system --quiet; then
            log_success "Installed: ${packages[*]}"
        else
            log_warning "UV installation failed for some packages, trying pip..."
            python -m pip install "${packages[@]}" --quiet || log_warning "Some packages failed to install"
        fi
    else
        # Use pip
        if python -m pip install "${packages[@]}" --quiet; then
            log_success "Installed: ${packages[*]}"
        else
            log_warning "Pip installation failed for some packages"
        fi
    fi
}

# Install core scientific packages
log_info "🔬 Installing core scientific packages..."
install_packages numpy scipy pandas matplotlib seaborn scikit-learn
install_packages jupyter jupyterlab ipywidgets
install_packages plotly requests beautifulsoup4

# Install development tools
log_info "🛠️ Installing development tools..."
install_packages pytest black flake8 mypy

# Install AI and modern tools from project dependencies
log_info "🤖 Installing AI and modern tools from project dependencies..."

# Initialize UV project and sync dependencies
if [ "$UV_AVAILABLE" = true ]; then
    log_info "🔄 Initializing UV project and syncing dependencies..."

    # Check if we're in a UV project, if not initialize one
    if [ ! -f "pyproject.toml" ]; then
        log_warning "pyproject.toml not found, skipping UV sync"
    else
        # Ensure UV cache directory exists
        mkdir -p /tmp/uv-cache

        # Create virtual environment if it doesn't exist
        if [ ! -d ".venv" ]; then
            log_info "📦 Creating UV virtual environment..."
            uv venv --quiet || log_warning "Failed to create virtual environment"
        fi

        # Run uv sync to install all project dependencies including marimo
        if uv sync --quiet; then
            log_success "Project dependencies installed successfully with uv sync"

            # Activate the virtual environment in shell profiles
            log_info "🔧 Activating virtual environment in shell profiles..."
            echo "source $(pwd)/.venv/bin/activate" >> ~/.bashrc
            echo "source $(pwd)/.venv/bin/activate" >> ~/.zshrc 2>/dev/null || true

            # Also activate for current session
            source .venv/bin/activate

            # Test if marimo is available
            if command -v marimo >/dev/null 2>&1; then
                log_success "✅ Marimo is now available!"
            else
                log_warning "⚠️  Marimo installation may have failed - you may need to run 'source .venv/bin/activate' manually"
            fi
        else
            log_warning "uv sync failed, trying alternative installation..."
            # Fallback: install marimo and other key dependencies directly
            install_packages marimo
        fi
    fi
else
    log_warning "UV not available, installing marimo with pip..."
    python -m pip install marimo --quiet || log_warning "Marimo installation failed"
fi

# Install AI assistant CLI tools
log_info "🤖 Installing AI assistant CLI tools..."

# Set up API keys from Codespaces secrets if available
setup_api_keys() {
    log_info "🔑 Setting up API keys..."

    # Check for Codespaces secrets and set them up
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\"" >> ~/.bashrc
        echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\"" >> ~/.zshrc 2>/dev/null || true
        log_success "OpenAI API key configured"
    else
        log_warning "OPENAI_API_KEY not found - AI features will be limited"
    fi

    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> ~/.bashrc
        echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> ~/.zshrc 2>/dev/null || true
        log_success "Anthropic API key configured"
    else
        log_warning "ANTHROPIC_API_KEY not found - Claude features will be limited"
    fi

    if [ -n "$GOOGLE_API_KEY" ]; then
        echo "export GOOGLE_API_KEY=\"$GOOGLE_API_KEY\"" >> ~/.bashrc
        echo "export GOOGLE_API_KEY=\"$GOOGLE_API_KEY\"" >> ~/.zshrc 2>/dev/null || true
        log_success "Google API key configured"
    else
        log_warning "GOOGLE_API_KEY not found - Gemini features will be limited"
    fi

    # Create .env file for applications that need it
    cat > ~/.env << EOF
# API Keys (auto-generated from Codespaces secrets)
OPENAI_API_KEY=${OPENAI_API_KEY:-}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
EOF

    log_success "API keys setup complete"
    log_info "💡 To add API keys: GitHub repo → Settings → Secrets → Codespaces"
}

setup_api_keys

# Install Python-based AI SDKs for programmatic access
install_packages openai anthropic google-generativeai llm 2>/dev/null || log_warning "Some AI SDKs failed to install"

# Amp Code CLI (amp) install
log_info "🧩 Checking for Amp Code CLI (amp)..."

# Persist AMP_API_KEY if present
if [ -n "$AMP_API_KEY" ]; then
    echo "export AMP_API_KEY=\"$AMP_API_KEY\"" >> ~/.bashrc
    echo "export AMP_API_KEY=\"$AMP_API_KEY\"" >> ~/.zshrc 2>/dev/null || true
    log_success "AMP_API_KEY configured"
else
    log_warning "AMP_API_KEY not found - Amp Code features may be limited"
fi

install_amp_code_cli() {
    # Skip if already installed
    if command -v amp >/dev/null 2>&1; then
        log_success "Amp Code CLI already installed"
        return 0
    fi

    # Allow users to skip install explicitly
    if [ "${AMP_SKIP_INSTALL:-0}" = "1" ]; then
        log_warning "Skipping Amp Code CLI install due to AMP_SKIP_INSTALL=1"
        return 0
    fi

    log_info "⬇️  Installing Amp Code CLI (see https://ampcode.com/manual for details)"

    local installed=false

    # Preferred: official installer script (if reachable)
    if command -v curl >/dev/null 2>&1; then
        if curl -fsSL https://ampcode.com/install.sh | bash >/dev/null 2>&1; then
            installed=true
        fi
    fi

    # Verify install
    if [ "$installed" = true ] && command -v amp >/dev/null 2>&1; then
        log_success "Amp Code CLI installed"
        return 0
    fi

    # Fallback notice
    log_warning "Amp Code CLI install not completed automatically."
    log_info "Follow manual steps inside the container:"
    echo "  1) Open https://ampcode.com/manual"
    echo "  2) Run the documented install command (e.g., curl | bash)"
    echo "  3) Verify with: amp --version"
    return 0
}

install_amp_code_cli || true

# Try to install common AI CLI tools (if npm available)
if command -v npm >/dev/null 2>&1; then
    # Install Google Gemini CLI
    npm install -g @google/gemini-cli 2>/dev/null || log_warning "Gemini CLI installation failed"

    # Install Claude Code CLI
    npm install -g @anthropic-ai/claude-code 2>/dev/null || log_warning "Claude Code CLI installation failed"

    # Install OpenAI Codex CLI
    npm install -g @openai/codex 2>/dev/null || log_warning "OpenAI Codex CLI installation failed"

    # Install OpenCode AI
    npm install -g opencode-ai 2>/dev/null || log_warning "OpenCode AI installation failed"
else
    log_warning "npm not available, installing fewer AI CLI tools"
fi

# Install visualization packages
log_info "📊 Installing visualization packages..."
install_packages altair bokeh graphviz

# Install web development packages
log_info "🌐 Installing web development packages..."
install_packages fastapi streamlit gradio dash flask

# Configure Jupyter
log_info "📓 Configuring Jupyter..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.allow_remote_access = True
c.LabApp.default_url = '/lab'
EOF

log_success "✅ Scientific Programming Environment setup complete!"
log_info ""
log_info "🎓 Next steps:"
echo "1. Start Jupyter Lab: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "2. Try Marimo: marimo edit"
echo "3. Check the templates in ~/templates/"
echo "4. Read README_devcontainer.md for more info"
echo "5. Set up AI tools: Check TROUBLESHOOTING.md for API key setup"
echo "6. Try AI helper: python ~/ai_helper.py 'Explain machine learning'"
echo ""
log_success "🔬 Happy scientific programming!"

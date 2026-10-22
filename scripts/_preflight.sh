# Shared preflight for the standalone runners. Sourced, not executed.
# Makes a run depend on nothing but the repo and a working Ollama install.

preflight() {
  local need_ollama="${1:-yes}"

  # 1. project root, resolved from this file's location
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$ROOT"

  # 2. the venv, used directly so no `activate` and no inherited PATH is needed
  PY="$ROOT/.venv/bin/python"
  if [ ! -x "$PY" ]; then
    echo "ERROR: no virtualenv at $PY"
    echo "  Create it from the project root:"
    echo "    uv venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -r requirements.txt"
    return 1
  fi
  export VIRTUAL_ENV="$ROOT/.venv"
  export PATH="$ROOT/.venv/bin:$PATH"

  # 3. .env -> SEED, model tags, PYTORCH_ENABLE_MPS_FALLBACK, OLLAMA_*
  if [ ! -f "$ROOT/.env" ]; then
    echo "ERROR: $ROOT/.env is missing (it carries SEED, SLM_TAG and the Ollama settings)."
    return 1
  fi
  set -a; . "$ROOT/.env"; set +a
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

  # 4. python deps actually import
  if ! "$PY" -c 'import torch, torch_geometric, networkx, sentence_transformers, psutil' 2>/dev/null; then
    echo "ERROR: the venv is missing dependencies. From the project root:"
    echo "    VIRTUAL_ENV=.venv uv pip install -r requirements.txt"
    return 1
  fi

  # 5. Ollama reachable, started if necessary
  if [ "$need_ollama" = "yes" ]; then
    if ! command -v ollama >/dev/null 2>&1; then
      echo "ERROR: the \`ollama\` command is not on PATH. Install Ollama and retry."
      return 1
    fi
    if ! curl -fsS --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/version" >/dev/null 2>&1; then
      echo "  ollama server not responding; starting it..."
      nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
      for _ in $(seq 1 20); do
        sleep 1
        curl -fsS --max-time 2 "${OLLAMA_HOST:-http://localhost:11434}/api/version" >/dev/null 2>&1 && break
      done
      if ! curl -fsS --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/version" >/dev/null 2>&1; then
        echo "ERROR: could not start the ollama server. See /tmp/ollama_serve.log"
        return 1
      fi
    fi
  fi

  # 6. power and memory context, for the record
  if [ "$(pmset -g batt 2>/dev/null | head -1 | grep -c 'AC Power')" -eq 0 ]; then
    echo "  WARNING: on battery. agent.md Section 2 wants long runs on AC power."
  fi
  "$PY" -c 'import psutil; v=psutil.virtual_memory(); print(f"  swap {int(psutil.swap_memory().used/1048576)} MB | RAM available {int(v.available/1048576)} MB")'
  return 0
}

require_files() {
  local missing=0
  for f in "$@"; do
    [ -e "$ROOT/$f" ] || { echo "ERROR: required input missing: $f"; missing=1; }
  done
  [ "$missing" -eq 0 ] || { echo "  Earlier phases must finish before this run."; return 1; }
  return 0
}

#!/bin/bash
#
# Runs bottomup.py on given SSH hosts.
#
# Example invocation:
#   `scripts/bottomup_ssh.sh host1 host2 -- --moves-only --size 8`
#
# This script assumes the first given host's filesystem is shared by all (e.g., home
# directory is on NFS). Additionally, the script does not install the dependencies
# required for evaluation scripts; only the dependencies requried by bottomup.py are
# installed.

set -e

REMOTE_DEST="~/morello_bottomup"
WORKER_NICE=10
NWORKERS=-1  # per host
TMUX_SESSION_NAME=dask
MEM_LIMIT=7G

# Parse CLI args.
stopping=false
while getopts "sn::m::" option; do
   case $option in
     s) stopping=true;;
     n) NWORKERS="$OPTARG";;
     m) MEM_LIMIT="$OPTARG";;
     *) echo "Invalid option: -$OPTARG" >&2
        exit 1;;
   esac
done
shift "$((OPTIND - 1))"

# Read target hosts from the command line. The first will run the script
# and scheduler.
declare -a ALL_HOSTS
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; break;;
    *) ALL_HOSTS+=("$1"); shift;;
  esac
done
MAIN_HOST="${ALL_HOSTS[0]}"
declare -a OTHER_HOSTS=( "${ALL_HOSTS[@]:1}" )

# Read the arguments to forward.
declare -a EXTRA_ARGS=( "$@" )

# If -s was passed, just stop the tmux sessions.
if [ "$stopping" = true ]; then
    for d in "${ALL_HOSTS[@]}"; do
        echo "Stopping on $d"
        ssh "$d" "tmux kill-session -t '$TMUX_SESSION_NAME'" || true
        ssh "$d" "killall redis-server" || true
    done
    exit 0
fi

# Install poetry
if ssh "$MAIN_HOST" "stat ~/.local/bin/poetry" > /dev/null 2>&1; then
    echo "Poetry already installed on main host."
else
    echo "Installing poetry on main host."
    ssh "${MAIN_HOST}" "curl -sSL https://install.python-poetry.org | python3 -"
fi

# Copy .gitignore so it's is respected at the destination during the rsync.
ssh "${MAIN_HOST}" "mkdir -p \"$REMOTE_DEST\""
scp .gitignore "$MAIN_HOST:$REMOTE_DEST/.gitignore"

# Sync project directory.
rsync -vhra ./ "$MAIN_HOST:$REMOTE_DEST" --include='**.gitignore' \
  --exclude='/.git' --filter=':- .gitignore' --delete-after

# Install Python dependencies
ssh "${MAIN_HOST}" "cd $REMOTE_DEST && ~/.local/bin/poetry install --without=evaluation"

# TODO: Grab REDIS_SERVER_PATH from the destination host environment.
REDIS_SERVER_PATH=/homes/gws/kaufmans/local/bin/redis-server
REDIS_PORT=7771
REDIS_URL="redis://127.0.0.1:$REDIS_PORT/"

EXP_BIT="export DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL=20m DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=480 REDIS_URL=$REDIS_URL"

# Run dask-worker, dash-scheduler, and script on main host.
echo "Starting script, scheduler, and worker on ${MAIN_HOST}."
ssh "${MAIN_HOST}" "cd $REMOTE_DEST && \
     tmux new-session -d -s '$TMUX_SESSION_NAME' \
    '$EXP_BIT; poetry run dask scheduler --host $MAIN_HOST' ';' \
    split '$REDIS_SERVER_PATH --port $REDIS_PORT --bind 127.0.0.1' ';' \
    split 'sleep 10; $EXP_BIT; poetry run nice -n $WORKER_NICE dask worker --memory-limit=$MEM_LIMIT --nworkers=$NWORKERS --nthreads 1 $MAIN_HOST:8786' ';' \
    split 'sleep 5; $EXP_BIT; poetry run python -m morello.search.bottomup --scheduler $MAIN_HOST:8786 ${EXTRA_ARGS[*]}' ';' \
    setw remain-on-exit on ';'"

# Run dask-workers on the other hosts.
for d in "${OTHER_HOSTS[@]}"; do
  echo "Starting worker on $d."
  ssh "$d" "cd $REMOTE_DEST && \
      tmux new-session -d -s '$TMUX_SESSION_NAME' \
      'sleep 5; $EXP_BIT; poetry run nice -n $WORKER_NICE dask worker --name $d --memory-limit=$MEM_LIMIT --nworkers=$NWORKERS --nthreads 1 $MAIN_HOST:8786' ';' \
      setw remain-on-exit on ';'"
done
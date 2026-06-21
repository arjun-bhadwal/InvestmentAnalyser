#!/bin/bash
# ============================================================
#  Investment Analyser — startup / restart script
#
#  Usage:
#    ./start.sh          — restart everything (server + tunnel)
#    ./start.sh --server — restart server only (e.g. after .env change)
# ============================================================

set -euo pipefail

PLIST_SERVER=~/Library/LaunchAgents/com.investmentanalyser.server.plist
PLIST_TUNNEL=~/Library/LaunchAgents/com.investmentanalyser.tunnel.plist
TUNNEL_LOG=/tmp/cloudflared.err

SERVER_ONLY=false
if [[ "${1:-}" == "--server" ]]; then
    SERVER_ONLY=true
fi

echo ""
echo "=================================================="
echo "  Investment Analyser — Starting Up"
echo "=================================================="
echo ""

DOMAIN="gui/$(id -u)"
LABEL_SERVER="com.investmentanalyser.server"
LABEL_TUNNEL="com.investmentanalyser.tunnel"

# Bootstrap if not already registered, then kick; if already registered just kick
restart_service() {
    local label="$1" plist="$2"
    if ! launchctl print "$DOMAIN/$label" &>/dev/null; then
        launchctl bootstrap "$DOMAIN" "$plist"
    fi
    launchctl kickstart -kp "$DOMAIN/$label"
}

# ── 1. Restart MCP server ─────────────────────────────────
echo "▶  Restarting MCP server..."
restart_service "$LABEL_SERVER" "$PLIST_SERVER"

# ── 2. Restart tunnel (unless --server flag used) ─────────
if [ "$SERVER_ONLY" = false ]; then
    echo "▶  Restarting Cloudflare tunnel..."
    > "$TUNNEL_LOG"   # clear log so new URL is detected cleanly
    restart_service "$LABEL_TUNNEL" "$PLIST_TUNNEL"
fi

# ── 3. Wait for server ────────────────────────────────────
echo "▶  Waiting for server on localhost:8000..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null http://localhost:8000/mcp 2>/dev/null; then
        break
    fi
    sleep 1
done

# ── 4. Results ────────────────────────────────────────────
echo ""
echo "=================================================="

if curl -s -o /dev/null http://localhost:8000/mcp 2>/dev/null; then
    echo "  ✅  MCP server:     running  (desktop auto-connected)"
else
    echo "  ❌  MCP server:     NOT running — check /tmp/investmentanalyser.err"
fi

if [ "$SERVER_ONLY" = true ]; then
    echo "  ℹ️   Tunnel:         not restarted (--server flag)"
    echo "=================================================="
    echo ""
    exit 0
fi

# ── 5. Get tunnel URL ─────────────────────────────────────
echo "▶  Waiting for tunnel URL..."
TUNNEL_URL=""
for i in $(seq 1 20); do
    TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1 || true)
    [ -n "$TUNNEL_URL" ] && break
    sleep 1
done

if [ -n "$TUNNEL_URL" ]; then
    MCP_ENDPOINT="${TUNNEL_URL}/mcp"
    echo "  ✅  Tunnel:         connected"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │  📱  MOBILE URL (paste into Claude connector):              │"
    echo "  │      $MCP_ENDPOINT"
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  → Claude mobile: Connectors → T212-Analyser → Remove"
    echo "    then Add custom connector → paste URL above"
else
    echo "  ❌  Tunnel:         not detected — check /tmp/cloudflared.err"
fi

echo "=================================================="
echo ""

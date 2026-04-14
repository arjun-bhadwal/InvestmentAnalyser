# Investment Analyser — MCP Server

A remote MCP server that connects to the Trading 212 API and exposes your portfolio, market data, and financial news to Claude. Runs locally on your Mac, exposed via Cloudflare Tunnel.

## Tools

| Tool | Description | Example prompt |
|---|---|---|
| `get_portfolio` | Live positions: ticker, qty, avg price, current price, P&L | "Show me my portfolio" |
| `get_account_summary` | Total value, free cash, invested, unrealised P&L | "What's my account balance?" |
| `get_price` | Current price + day change for any ticker | "What's the price of TSLA?" |
| `get_news` | Last 5 headlines for a stock (via Finnhub) | "Any news on Apple?" |
| `get_market_snapshot` | FTSE 100, S&P 500, NASDAQ moves today | "How are markets doing?" |

---

## Prerequisites

- Python 3.11+
- [Homebrew](https://brew.sh)
- A free [Cloudflare account](https://dash.cloudflare.com/sign-up)
- [Trading 212](https://www.trading212.com/) Invest or ISA account — Settings → API → generate a **Practice** key to start
- [Finnhub](https://finnhub.io/register) free API key

---

## 1. Set up the Python environment

```bash
cd /path/to/InvestmentAnalyser

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
T212_API_KEY=your_practice_account_key
T212_MODE=demo
FINNHUB_API_KEY=your_finnhub_key
MCP_TOKEN=your_random_secret          # openssl rand -hex 32
```

---

## 3. Create a Cloudflare Tunnel

This gives you a stable public HTTPS URL with no open ports or port forwarding.

**Install cloudflared:**

```bash
brew install cloudflare/cloudflare/cloudflared
```

**Log in to Cloudflare:**

```bash
cloudflared tunnel login
```

A browser window opens — authorise with your Cloudflare account.

**Create a named tunnel:**

```bash
cloudflared tunnel create investment-analyser
```

This outputs a tunnel UUID, e.g. `a1b2c3d4-...`. Note it down.

**Create the tunnel config file:**

```bash
mkdir -p ~/.cloudflared
```

Create `~/.cloudflared/investment-analyser.yml`:

```yaml
tunnel: <your-tunnel-uuid>
credentials-file: /Users/<your-username>/.cloudflared/<your-tunnel-uuid>.json

ingress:
  - service: http://localhost:8000
```

Replace `<your-tunnel-uuid>` and `<your-username>` with real values.

**Get your stable tunnel URL:**

Your public URL is always: `https://<your-tunnel-uuid>.cfargotunnel.com`

No DNS setup needed — this URL is active as long as the tunnel is running.

---

## 4. Test it manually

In one terminal, start the server:

```bash
source .venv/bin/activate
python server.py
```

In another terminal, start the tunnel:

```bash
cloudflared tunnel --config ~/.cloudflared/investment-analyser.yml run
```

Smoke test:

```bash
curl -s -X POST https://<your-tunnel-uuid>.cfargotunnel.com/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_mcp_token" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python -m json.tool
```

You should see all 5 tools listed.

---

## 5. Auto-start on login (launchd)

This keeps both the server and the tunnel running automatically, restarting them if they crash.

**Create the server launch agent:**

Save as `~/Library/LaunchAgents/com.investmentanalyser.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.investmentanalyser.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/InvestmentAnalyser/.venv/bin/python</string>
        <string>/path/to/InvestmentAnalyser/server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/InvestmentAnalyser</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/investmentanalyser.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/investmentanalyser.err</string>
</dict>
</plist>
```

**Create the tunnel launch agent:**

Save as `~/Library/LaunchAgents/com.investmentanalyser.tunnel.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.investmentanalyser.tunnel</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/cloudflared</string>
        <string>tunnel</string>
        <string>--config</string>
        <string>/Users/<your-username>/.cloudflared/investment-analyser.yml</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/cloudflared.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/cloudflared.err</string>
</dict>
</plist>
```

**Load both agents:**

```bash
launchctl load ~/Library/LaunchAgents/com.investmentanalyser.server.plist
launchctl load ~/Library/LaunchAgents/com.investmentanalyser.tunnel.plist
```

They start immediately and will restart on crash or reboot.

**Useful commands:**

```bash
# Check status
launchctl list | grep investmentanalyser

# View logs
tail -f /tmp/investmentanalyser.log
tail -f /tmp/cloudflared.log

# Restart server
launchctl stop com.investmentanalyser.server   # launchd auto-restarts it

# Unload (stop permanently)
launchctl unload ~/Library/LaunchAgents/com.investmentanalyser.server.plist
launchctl unload ~/Library/LaunchAgents/com.investmentanalyser.tunnel.plist
```

---

## 6. Register as a Claude Connector

1. Go to **claude.ai** → click your avatar (bottom-left) → **Settings** → **Connectors**
2. Click **Add connector**
3. Paste your tunnel URL: `https://<your-tunnel-uuid>.cfargotunnel.com/mcp`
4. Under authentication, select **Bearer token** and paste your `MCP_TOKEN`
5. Name it **Investment Analyser** and save

On **Claude mobile** (iOS/Android): connectors added via claude.ai sync automatically.

---

## 7. Switching to your Live account

When you're ready to use real money data:

1. Generate a **Live** API key in Trading 212 → Settings → API
2. Update `.env`: set `T212_API_KEY` to the live key and `T212_MODE=live`
3. Restart the server: `launchctl stop com.investmentanalyser.server`

Optionally whitelist your Mac's outbound IP in Trading 212's API settings (Settings → API → your key → IP restrictions). Your outbound IP: `curl ifconfig.me`

---

## Finding your cloudflared path

If `cloudflared` was installed via Homebrew on Apple Silicon, the binary may be at `/opt/homebrew/bin/cloudflared` instead of `/usr/local/bin/cloudflared`. Check with:

```bash
which cloudflared
```

Use that path in the launchd plist.

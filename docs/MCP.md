# Agent-R — MCP server

`agentr mcp` starts a **stdio MCP server** (JSON-RPC 2.0, one response per
request line). It lets any MCP-capable LLM client — Claude Desktop, IDE
agents, custom orchestrators — read and write the shared Agent-R error
memory without speaking HTTP.

- Protocol version: `2024-11-05`; capabilities: `tools`.
- State: same snapshot as the CLI and server (`AGENT_R_STATE_PATH`, default
  `~/.agentr/state.json`) — one memory, three frontends.
- Implementation: `src/bin/agentr.rs` (`mcp_serve`), no external MCP SDK
  needed.

## Tools

| Tool | Input schema | Purpose |
|------|--------------|---------|
| `agentr_check` | `operation` (required), `situation`, `tags: [string]` | Preflight: which registered lessons apply to this operation before running it. Response includes `proceed`, `warnings`, `precautions`, `top_rules`, `suggestion`, `veto_reason`, `w_content`. If `situation` is given, the hybrid content signal is used. |
| `agentr_remember` | `operation` (required), `what` (required, short concrete lesson), `why`, `prevention`, `severity` ∈ `warning \| error \| critical`, `category` ∈ `circular \| drift \| fabrication \| assumption \| skipped \| timeout \| corruption \| recurring \| other`, `tags: [string]` | Record an error/lesson for future preflights (auto-dedup by fingerprint). |
| `agentr_outcome` | `outcome` ∈ `hit \| miss` (required) | Self-tuning feedback. `hit`: the retrieved lesson applied but the task still failed (strengthen the signal). `miss`: the signal did not apply (decay it). |
| `agentr_stats` | — | Memory and self-tuning counters (incidents, preflights, blocks, per-category counts, tuning hits/misses). |

The intended loop for an agent: **check** before acting → act →
**remember** on failure → **outcome** after a preflight-influenced run →
**stats** to introspect.

## Example session (raw JSON-RPC over stdio)

```jsonc
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"agentr","version":"1.0.0"}}}

{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"agentr_check", …}, …]}}

{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"agentr_check","arguments":{"operation":"deploy_edge","situation":"about to curl the edge API","tags":["deploy"]}}}
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"proceed\":true,…}"}]}}
```

## Client configuration

Claude Desktop (`claude_desktop_config.json`) — point the command at the
built binary and, if you want it to share state with a running server, set
`AGENT_R_STATE_PATH` for both:

```json
{
  "mcpServers": {
    "agentr": {
      "command": "/path/to/agent-r/target/release/agentr",
      "args": ["mcp"],
      "env": { "AGENT_R_STATE_PATH": "/home/you/.agentr/state.json" }
    }
  }
}
```

Generic MCP client config (same shape): command `agentr`, args `["mcp"]`.

> **Not to be confused**: `scripts/mcp_server.py` (+ `scripts/m2mctl.py`) is
> the repo's *development control plane* (build/test/serve/bench helpers),
> a FastMCP wrapper for maintainers — it is unrelated to the agent memory
> exposed by `agentr mcp`.

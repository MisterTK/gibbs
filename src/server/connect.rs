//! Ready-to-paste setups that point coding agents at this server (banner, TUI `/connect`,
//! `/agent`).

use std::net::SocketAddr;

/// Agent CLIs `/agent` launches.
pub const AGENTS: [&str; 4] = ["claude", "codex", "opencode", "aider"];

/// The address clients dial: a wildcard listen address becomes loopback.
pub fn client_addr(addr: SocketAddr) -> SocketAddr {
    if addr.ip().is_unspecified() {
        SocketAddr::from(([127, 0, 0, 1], addr.port()))
    } else {
        addr
    }
}

/// Single-quoted for POSIX shells.
pub(crate) fn q(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Shell command that starts `agent` against the server at `addr`, asking for `model`.
pub fn agent_command(agent: &str, addr: SocketAddr, model: &str) -> Result<String, String> {
    let a = client_addr(addr);
    Ok(match agent {
        "claude" => format!(
            "ANTHROPIC_BASE_URL=http://{a} ANTHROPIC_AUTH_TOKEN=gibbs ANTHROPIC_MODEL={} claude",
            q(model)
        ),
        "codex" => format!(
            "codex -c model_provider=gibbs -c {} -m {}",
            q(&format!(
                "model_providers.gibbs={{name=\"gibbs\",base_url=\"http://{a}/v1\",wire_api=\"responses\"}}"
            )),
            q(model)
        ),
        "opencode" => {
            let config = serde_json::json!({"provider": {"gibbs": {
                "npm": "@ai-sdk/openai-compatible", "name": "gibbs",
                "options": {"baseURL": format!("http://{a}/v1")},
                "models": {model: {}}}}});
            format!(
                "OPENCODE_CONFIG_CONTENT={} opencode -m {}",
                q(&config.to_string()),
                q(&format!("gibbs/{model}"))
            )
        }
        "aider" => format!(
            "OPENAI_API_BASE=http://{a}/v1 OPENAI_API_KEY=gibbs aider --model {}",
            q(&format!("openai/{model}"))
        ),
        other => {
            return Err(format!(
                "unknown agent '{other}'; use one of: {}",
                AGENTS.join(", ")
            ))
        }
    })
}

/// Every client setup, for `/connect`.
pub fn setup_text(addr: SocketAddr, model: &str) -> String {
    let a = client_addr(addr);
    let cmd = |agent| agent_command(agent, addr, model).expect("known agent");
    format!(
        "Point an agent at http://{a} (any model name serves the loaded model; any API key works)

Claude Code (Anthropic /v1/messages):
  {claude}

Codex (OpenAI /v1/responses), ~/.codex/config.toml:
  model = \"{model}\"
  model_provider = \"gibbs\"
  [model_providers.gibbs]
  name = \"gibbs\"
  base_url = \"http://{a}/v1\"
  wire_api = \"responses\"

opencode / Continue / any OpenAI client (/v1/chat/completions): base URL http://{a}/v1
  {opencode}

aider:
  {aider}

{compat_client} clients (/api/chat, /api/generate):
  {compat_env}={a}

Launch one in a new window: /agent {agents}",
        claude = cmd("claude"),
        opencode = cmd("opencode"),
        aider = cmd("aider"),
        agents = AGENTS.join("|"),
        compat_client = crate::arch::compat()["ndjson_api"]["client_name"].as_str().expect("compat.ndjson_api.client_name"),
        compat_env = crate::arch::compat()["ndjson_api"]["host_env"].as_str().expect("compat.ndjson_api.host_env"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_dial_loopback_and_quote_model() {
        let addr = SocketAddr::from(([0, 0, 0, 0], 9170));
        let c = agent_command("claude", addr, "it's/m").unwrap();
        assert!(
            c.starts_with("ANTHROPIC_BASE_URL=http://127.0.0.1:9170 "),
            "{c}"
        );
        assert!(c.contains("ANTHROPIC_MODEL='it'\\''s/m'"), "{c}");
        assert!(agent_command("codex", addr, "m")
            .unwrap()
            .contains("base_url=\"http://127.0.0.1:9170/v1\""));
        assert!(agent_command("vim", addr, "m")
            .unwrap_err()
            .contains("claude, codex"));
    }
}

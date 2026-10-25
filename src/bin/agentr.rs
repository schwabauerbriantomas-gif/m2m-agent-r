//! CLI de Agent-R: uso diario directo con la lib (latencia de proceso ~ms,
//! cero red) + `serve` para levantar el server HTTP cuando hace falta.
//! Estado persistente compartido vía AGENT_R_STATE_PATH (default:
//! ~/.agentr/state.json).
//!
//! Comandos:
//!   agentr remember <op> <what> [--why W] [--severity S] [--category C] [--tags t1,t2] [--prevention P]
//!   agentr check <op> [--situation S] [--tags t1,t2]     → preflight híbrido (¿qué lecciones aplican?)
//!   agentr validate --op OP [--output O]                  → validate N0-N3 del output
//!   agentr show <id> | deprecate <id> <reason> | stats | flush
//!   agentr mcp                                            → servidor MCP stdio (JSON-RPC 2.0)
//!   agentr serve [--port N]                               → server HTTP (default 8199, 127.0.0.1)
//!
//! Formato de salida: texto para humano, --json para pipes/scripts.

use agent_r::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use std::path::PathBuf;

fn state_path() -> PathBuf {
    std::env::var("AGENT_R_STATE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".agentr").join("state.json")
        })
}

fn severity_of(s: &str) -> ErrorSeverity {
    match s.to_lowercase().as_str() {
        "warning" | "w" => ErrorSeverity::Warning,
        "critical" | "crit" | "c" => ErrorSeverity::Critical,
        _ => ErrorSeverity::Error,
    }
}

fn category_of(s: &str) -> ErrorCategory {
    match s.to_lowercase().as_str() {
        "circular" | "circularreasoning" => ErrorCategory::CircularReasoning,
        "drift" | "outputdrift" => ErrorCategory::OutputDrift,
        "fabrication" | "fab" => ErrorCategory::Fabrication,
        "assumption" | "badassumption" => ErrorCategory::BadAssumption,
        "skipped" | "skippedvalidation" => ErrorCategory::SkippedValidation,
        "timeout" => ErrorCategory::Timeout,
        "corruption" | "statecorruption" => ErrorCategory::StateCorruption,
        "recurring" | "recurringpattern" => ErrorCategory::RecurringPattern,
        _ => ErrorCategory::Other,
    }
}

fn open_agent() -> AgentR {
    let mut agent = AgentR::with_state_path(state_path());
    // CLI de uso diario: si el usuario pidió AGENT_R_NO_SEED, con_state_path
    // ya no carga semillas; nada extra que hacer acá.
    agent
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_out = args.iter().any(|a| a == "--json");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let get = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        args.get(i + 1).cloned()
    };

    let cmd = pos.first().map(|s| s.as_str()).unwrap_or("help");
    let t0 = std::time::Instant::now();

    match cmd {
        "remember" => {
            let op = match pos.get(1) {
                Some(o) => o.to_string(),
                None => die("uso: agentr remember <op> <what> [--why W] [--severity S] [--category C] [--tags a,b] [--prevention P]"),
            };
            let what = match pos.get(2) {
                Some(w) => w.to_string(),
                None => {
                    die("falta <what>: la lección corta y concreta (es lo que matchea después)")
                }
            };
            let why = get("--why").unwrap_or_default();
            let sev = severity_of(&get("--severity").unwrap_or_else(|| "error".into()));
            let cat = category_of(&get("--category").unwrap_or_else(|| "other".into()));
            let prevention = get("--prevention").unwrap_or_default();
            let tags: Vec<String> = get("--tags")
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            let mut agent = open_agent();
            let id = agent.record_error(cat, sev, &op, &what, &why, &tag_refs, &prevention);
            let _ = agent.flush();
            out(
                json_out,
                &serde_json::json!({"id": id, "ms": elapsed(t0)}),
                &format!("recordado id={id} ({:.1} ms)", elapsed(t0)),
            );
        }
        "check" => {
            let op = match pos.get(1) {
                Some(o) => o.to_string(),
                None => die("uso: agentr check <op> [--situation S] [--tags a,b]"),
            };
            let situation = get("--situation").unwrap_or_default();
            let tags: Vec<String> = get("--tags")
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            let mut agent = open_agent();
            let pf = if situation.is_empty() {
                agent.preflight_v2(&op, &tag_refs)
            } else {
                agent.preflight_v2_hybrid(&op, &tag_refs, &situation)
            };
            let (hits, misses, w) = agent.tuning_stats();
            if json_out {
                out(
                    true,
                    &serde_json::json!({
                        "proceed": pf.proceed, "warnings": pf.warnings,
                        "precautions": pf.precautions, "top_rules": pf.top_rules,
                        "suggestion": pf.suggestion, "veto_reason": pf.veto_reason,
                        "w_content": w, "hits": hits, "misses": misses, "ms": elapsed(t0),
                    }),
                    "",
                );
                return;
            }
            let mut s = String::new();
            if !pf.proceed {
                s.push_str(&format!(
                    "⛔ VETO: {}\n   → {}\n",
                    pf.veto_reason
                        .as_deref()
                        .unwrap_or("regla crítica recurrente"),
                    pf.suggestion.as_deref().unwrap_or("")
                ));
            } else if !pf.warnings.is_empty() {
                s.push_str("⚠️  advertencias:\n");
                for w in &pf.warnings {
                    s.push_str(&format!("   • {w}\n"));
                }
            }
            if pf.top_rules.is_empty() && pf.warnings.is_empty() && pf.proceed {
                s.push_str("✓ sin lecciones previas para esta operación\n");
            }
            for r in &pf.top_rules {
                s.push_str(&format!(
                    "  [{}] (score {:.1}) {}\n",
                    r.id,
                    r.score,
                    truncate(&r.what, 90)
                ));
            }
            s.push_str(&format!("  ({:.1} ms)", elapsed(t0)));
            println!("{s}");
        }
        "feedback" => {
            // outcome del self-tuning: la task falló (hit) o completó (miss)
            let outcome = pos.get(1).map(|s| s.as_str()).unwrap_or("");
            match outcome {
                "hit" => {
                    let mut agent = open_agent();
                    let (_, w) = agent.tuning_feedback(Some(1), Some(1));
                    let _ = agent.flush();
                    println!("W={w:.3} ({:.1} ms)", elapsed(t0));
                }
                "miss" => {
                    let mut agent = open_agent();
                    let (_, w) = agent.tuning_feedback(Some(1), None);
                    let _ = agent.flush();
                    println!("W={w:.3} ({:.1} ms)", elapsed(t0));
                }
                _ => die("uso: agentr feedback hit|miss (hit = la lección aplicó y fallaste igual; miss = la señal no aplicaba)"),
            }
        }
        "deprecate" => {
            let id: u64 = pos
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| die("uso: agentr deprecate <id> <reason>"));
            let reason: String = match pos.get(2) {
                Some(r) => (*r).clone(),
                None => "manual".to_string(),
            };
            let mut agent = open_agent();
            let ok = agent.deprecate_error(id, &reason);
            let _ = agent.flush();
            out(
                json_out,
                &serde_json::json!({"ok": ok, "ms": elapsed(t0)}),
                &format!("deprecado {id}: {ok}"),
            );
        }
        "stats" => {
            let agent = open_agent();
            let st = agent.stats();
            let (h, m, w) = agent.tuning_stats();
            out(json_out, &serde_json::json!({
                "recorded": st.total_errors_recorded, "unique": st.unique_incidents,
                "preflights": st.total_preflights, "blocked": st.prefights_blocked,
                "tuning": {"hits": h, "misses": m, "w_content": w},
                "state": state_path().display().to_string(), "ms": elapsed(t0),
            }), &format!(
                "recordados {} | únicos {} | prefights {} ({} bloqueados) | W={w:.3} hits={h} misses={m} | {}",
                st.total_errors_recorded, st.unique_incidents, st.total_preflights,
                st.prefights_blocked, state_path().display()
            ));
        }
        "flush" => {
            let agent = open_agent();
            match agent.flush() {
                Ok(()) => out(json_out, &serde_json::json!({"ok": true}), "flush ok"),
                Err(e) => die(&format!("flush: {e}")),
            }
        }
        "path" => println!("{}", state_path().display()),
        "mcp" => mcp_serve(),
        "serve" => {
            // Server HTTP (migrado del CLI viejo): mismo run_server que hoy,
            // escucha en 127.0.0.1. Estado Agent-R vía AGENT_R_STATE_PATH.
            let port: u16 = match get("--port") {
                Some(p) => p
                    .parse()
                    .unwrap_or_else(|_| die("puerto inválido (uso: agentr serve --port N)")),
                None => 8199,
            };
            println!("[agentr] server HTTP en 127.0.0.1:{port} (Ctrl-C para cortar)");
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async {
                if let Err(e) = agent_r::api_server::run_server(port).await {
                    eprintln!("[agentr] server error: {e}");
                    std::process::exit(1);
                }
            });
        }
        _ => {
            println!("agentr — memoria de errores (CLI nativo, sin server)");
            println!();
            println!("  agentr remember <op> <what> [--why W] [--severity warning|error|critical]");
            println!("       [--category circular|drift|fabrication|assumption|skipped|timeout|corruption|recurring|other]");
            println!("       [--tags a,b] [--prevention P]");
            println!("  agentr check <op> [--situation S] [--tags a,b] [--json]");
            println!("  agentr feedback hit|miss        # outcome para el self-tuning");
            println!("  agentr deprecate <id> <reason>");
            println!("  agentr stats | flush | path");
            println!("  agentr mcp                      # servidor MCP stdio para agentes");
            println!("  agentr serve [--port N]         # server HTTP (default 8199, 127.0.0.1)");
            println!();
            println!("Estado: $AGENT_R_STATE_PATH o ~/.agentr/state.json");
        }
    }
}

fn elapsed(t0: std::time::Instant) -> f64 {
    (t0.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

fn out(json: bool, v: &serde_json::Value, text: &str) {
    if json {
        println!("{}", serde_json::to_string_pretty(v).unwrap());
    } else if !text.is_empty() {
        println!("{text}");
    }
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

// ── MCP stdio (JSON-RPC 2.0, una respuesta por línea de request) ─────────

fn mcp_serve() -> ! {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut agent = open_agent();
    eprintln!(
        "agentr MCP listo (stdio, estado: {})",
        state_path().display()
    );
    loop {
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            std::process::exit(0);
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = req.get("id").cloned().unwrap_or(serde_json::json!(0));
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let result = match method {
            "initialize" => serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agentr", "version": "1.0.0"}
            }),
            "notifications/initialized" => continue,
            "tools/list" => serde_json::json!({"tools": [
                {"name": "agentr_check", "description": "Preflight: qué lecciones registradas aplican a una operación antes de ejecutarla. Vetos y warnings incluidos.",
                 "inputSchema": {"type": "object", "properties": {
                    "operation": {"type": "string"},
                    "situation": {"type": "string", "description": "Estado actual (obs, comando, stderr). Activa la señal de contenido."},
                    "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["operation"]}},
                {"name": "agentr_remember", "description": "Registrar un error/lección para futuros preflights (dedup automático por fingerprint).",
                 "inputSchema": {"type": "object", "properties": {
                    "operation": {"type": "string"},
                    "what": {"type": "string", "description": "Lección corta y concreta"},
                    "why": {"type": "string"},
                    "prevention": {"type": "string"},
                    "severity": {"type": "string", "enum": ["warning", "error", "critical"]},
                    "category": {"type": "string", "enum": ["circular","drift","fabrication","assumption","skipped","timeout","corruption","recurring","other"]},
                    "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["operation", "what"]}},
                {"name": "agentr_outcome", "description": "Feedback del self-tuning: 'hit' (la lección aplicó y el task falló igual) o 'miss' (la señal no aplicaba).",
                 "inputSchema": {"type": "object", "properties": {
                    "outcome": {"type": "string", "enum": ["hit", "miss"]}},
                    "required": ["outcome"]}},
                {"name": "agentr_stats", "description": "Contadores de memoria y del self-tuning.",
                 "inputSchema": {"type": "object", "properties": {}}}
            ]}),
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or_default();
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let a = params.get("arguments").cloned().unwrap_or_default();
                let text = match name {
                    "agentr_check" => {
                        let op = a.get("operation").and_then(|o| o.as_str()).unwrap_or("");
                        let sit = a.get("situation").and_then(|o| o.as_str()).unwrap_or("");
                        let tags: Vec<String> = a
                            .get("tags")
                            .and_then(|t| t.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                        if op.is_empty() {
                            err_text("falta operation")
                        } else {
                            let pf = if sit.is_empty() {
                                agent.preflight_v2(op, &tag_refs)
                            } else {
                                agent.preflight_v2_hybrid(op, &tag_refs, sit)
                            };
                            let (_, _, w) = agent.tuning_stats();
                            serde_json::to_string(&serde_json::json!({
                                "proceed": pf.proceed, "warnings": pf.warnings,
                                "precautions": pf.precautions, "top_rules": pf.top_rules,
                                "suggestion": pf.suggestion, "veto_reason": pf.veto_reason,
                                "w_content": w,
                            }))
                            .unwrap_or_else(|_| err_text("serialize"))
                        }
                    }
                    "agentr_remember" => {
                        let op = a.get("operation").and_then(|o| o.as_str()).unwrap_or("");
                        let what = a.get("what").and_then(|o| o.as_str()).unwrap_or("");
                        if op.is_empty() || what.is_empty() {
                            err_text("falta operation/what")
                        } else {
                            let sev = severity_of(
                                a.get("severity")
                                    .and_then(|o| o.as_str())
                                    .unwrap_or("error"),
                            );
                            let cat = category_of(
                                a.get("category")
                                    .and_then(|o| o.as_str())
                                    .unwrap_or("other"),
                            );
                            let tags: Vec<String> = a
                                .get("tags")
                                .and_then(|t| t.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default();
                            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                            let id = agent.record_error(
                                cat,
                                sev,
                                op,
                                what,
                                a.get("why").and_then(|o| o.as_str()).unwrap_or(""),
                                &tag_refs,
                                a.get("prevention").and_then(|o| o.as_str()).unwrap_or(""),
                            );
                            let _ = agent.flush();
                            format!("{{\"id\": {id}}}")
                        }
                    }
                    "agentr_outcome" => {
                        let o = a.get("outcome").and_then(|o| o.as_str()).unwrap_or("");
                        let result: Result<(&'static str, f64), String> = match o {
                            "hit" => Ok(agent.tuning_feedback(Some(1), Some(1))),
                            "miss" => Ok(agent.tuning_feedback(Some(1), None)),
                            _ => Err("outcome debe ser hit|miss".to_string()),
                        };
                        match result {
                            Ok((out, w)) => {
                                let _ = agent.flush();
                                format!("{{\"outcome\": \"{out}\", \"w_content\": {w:.3}}}")
                            }
                            Err(e) => err_text(&e),
                        }
                    }
                    "agentr_stats" => {
                        let st = agent.stats();
                        let (h, m, w) = agent.tuning_stats();
                        serde_json::to_string(&serde_json::json!({
                            "recorded": st.total_errors_recorded, "unique": st.unique_incidents,
                            "preflights": st.total_preflights, "blocked": st.prefights_blocked,
                            "tuning": {"hits": h, "misses": m, "w_content": w},
                        }))
                        .unwrap_or_else(|_| err_text("serialize"))
                    }
                    _ => err_text(&format!("tool desconocida: {name}")),
                };
                serde_json::json!({
                    "content": [{"type": "text", "text": text}]
                })
            }
            other => {
                serde_json::json!({"error": {"code": -32601, "message": format!("método desconocido: {other}")}})
            }
        };
        let resp = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});
        println!("{}", serde_json::to_string(&resp).unwrap_or_default());
        let _ = std::io::stdout().flush();
    }
}

fn err_text(msg: &str) -> String {
    serde_json::json!({"error": msg}).to_string()
}

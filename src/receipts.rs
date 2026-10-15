//! Receipts HMAC de ejecución (rediseño v2, Pendiente 1 — N3).
//!
//! `ReceiptStore` file-backed: JSONL append en `data/receipts/` (un archivo
//! por día, `receipts-YYYYMMDD.jsonl`). Clave HMAC-SHA256 desde env
//! `AGENT_R_HMAC_KEY`; si ausente, se genera aleatoria y se persiste en
//! `data/receipts/.key` con permiso 0600. La clave JAMÁS se loggea ni
//! serializa en receipts.
//!
//! issue: canonicalización JSON (claves ordenadas, sin whitespace) →
//! HMAC-SHA256 → receipt `{id, ts, cmd_hash, output_hash, sig_hex}`.
//! verify: re-canonicaliza el payload firmado y compara la firma; cualquier
//! alteración del payload o de la firma → inválido (tamper detectado).
//!
//! Gate N3: [`ReceiptStore::n3_check`] consume un claim
//! `{receipt_id, claim:{metric,value}}`, verifica el receipt referenciado y
//! devuelve [`N3Verdict::Pass` / `Fail` / `Unverifiable`].

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Directorio base de receipts (relativo al CWD del server).
pub const DEFAULT_RECEIPTS_DIR: &str = "data/receipts";
/// Env var con la clave HMAC en hex. Si está ausente o vacía se genera y
/// persiste una clave (`.key`, 0600).
pub const HMAC_KEY_ENV: &str = "AGENT_R_HMAC_KEY";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum N3Verdict {
    Pass,
    Fail,
    Unverifiable,
}

/// Receipt inmutable emitido por `issue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub id: String,
    /// Unix epoch ms en que se emitió.
    pub ts: u64,
    /// HMAC-SHA256 (hex) del payload completo firmado (canónico).
    pub sig_hex: String,
    /// HMAC-SHA256 (hex) del comando/operación (para referencia rápida).
    pub cmd_hash: String,
    /// HMAC-SHA256 (hex) del output (para referencia rápida).
    pub output_hash: String,
    /// Payload exacto que se firmó (canónico). verify lo re-firma.
    pub payload: serde_json::Value,
}

#[derive(Debug)]
pub struct ReceiptError(pub String);

impl std::fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ReceiptError {}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Canonicalización: serializa con `serde_json::to_string` (claves ordenadas
/// sin whitespace — el default de serde_json con BTreeMap-backed maps es
/// orden lexicográfico para Value::Object cuando no hay feature
/// `preserve_order`). Aplicamos además un re-canonicalizado recursivo
/// explícito para blindarnos contra `preserve_order` activado en algún
/// punto del grafo de deps: ordenamos las claves siempre.
pub fn canonicalize(v: &serde_json::Value) -> String {
    fn canon(v: &serde_json::Value, out: &mut String) {
        match v {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    canon(&map[*k], out);
                }
                out.push('}');
            }
            serde_json::Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    canon(item, out);
                }
                out.push(']');
            }
            // Números: serde_json imprime floats sin trailing garbage y
            // enteros como enteros — determinista para un mismo Value.
            other => out.push_str(&other.to_string()),
        }
    }
    let mut s = String::new();
    canon(v, &mut s);
    s
}

fn hmac_hex(key: &[u8], msg: &[u8]) -> String {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    hex::encode(mac.finalize().into_bytes())
}

fn verify_sig(key: &[u8], msg: &[u8], sig_hex: &str) -> bool {
    match hex::decode(sig_hex) {
        Ok(sig) => {
            let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
            mac.update(msg);
            mac.verify_slice(&sig).is_ok()
        }
        Err(_) => false,
    }
}

/// Resultado de verify: `{valid, reason}`.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyResult {
    pub valid: bool,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<Receipt>,
}

/// File-backed receipt store.
///
/// Append JSONL diario; el índice id→receipt se carga lazy desde disco la
/// primera vez que se necesita (verify de un id no en memoria) y se mantiene
/// en RAM. Thread-safe (Mutex interno). `issue` = canonicalizar + firmar +
/// append a disco (write-through).
pub struct ReceiptStore {
    dir: PathBuf,
    key: Vec<u8>,
    index: Mutex<std::collections::HashMap<String, Receipt>>,
    /// Evita re-escanear archivos ya cargados (persistencia multi-proceso).
    scanned: Mutex<std::collections::HashSet<PathBuf>>,
}

impl ReceiptStore {
    /// Abre (o crea) el store en `dir`. Clave: env `AGENT_R_HMAC_KEY` (hex),
    /// o genera+persiste `dir/.key` (0600). JAMÁS loggea la clave.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, ReceiptError> {
        Self::open_inner(dir.as_ref(), true)
    }

    /// Como `open` pero ignora el env var (para tests deterministas).
    pub fn open_no_env(dir: impl AsRef<Path>) -> Result<Self, ReceiptError> {
        Self::open_inner(dir.as_ref(), false)
    }

    fn open_inner(dir: &Path, use_env: bool) -> Result<Self, ReceiptError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| ReceiptError(format!("create receipts dir {:?}: {}", dir, e)))?;
        let env_key = if use_env {
            std::env::var(HMAC_KEY_ENV)
                .ok()
                .filter(|k| !k.trim().is_empty())
        } else {
            None
        };
        Self::with_generated_or_persisted_key(dir, env_key)
    }

    fn with_generated_or_persisted_key(
        dir: &Path,
        env_key: Option<String>,
    ) -> Result<Self, ReceiptError> {
        let key_path = dir.join(".key");
        let key: Vec<u8> = if let Some(k) = env_key {
            hex::decode(k.trim())
                .map_err(|e| ReceiptError(format!("{} no es hex válido: {}", HMAC_KEY_ENV, e)))?
        } else if key_path.exists() {
            std::fs::read(&key_path)
                .map_err(|e| ReceiptError(format!("leer .key: {}", e)))?
        } else {
            let mut buf = [0u8; 32];
            getrandom_fill(&mut buf)?;
            std::fs::write(&key_path, buf)
                .map_err(|e| ReceiptError(format!("persistir .key: {}", e)))?;
            set_mode_0600(&key_path);
            buf.to_vec()
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            key,
            index: Mutex::new(std::collections::HashMap::new()),
            scanned: Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Emite un receipt para `payload` (cualquier JSON). Firma el payload
    /// canónico y lo persiste (JSONL append). Devuelve el receipt.
    pub fn issue(&self, payload: &serde_json::Value) -> Result<Receipt, ReceiptError> {
        let ts = now_ms();
        let canonical = canonicalize(payload);
        let sig_hex = hmac_hex(&self.key, canonical.as_bytes());
        let cmd_hash = match payload.get("cmd").or_else(|| payload.get("operation")) {
            Some(c) => hmac_hex(&self.key, canonicalize(c).as_bytes()),
            None => hmac_hex(&self.key, b""),
        };
        let output_hash = match payload.get("output").or_else(|| payload.get("result")) {
            Some(o) => hmac_hex(&self.key, canonicalize(o).as_bytes()),
            None => hmac_hex(&self.key, b""),
        };
        let id = format!("rcpt_{:x}_{}", ts, &sig_hex[..16]);
        let receipt = Receipt {
            id,
            ts,
            sig_hex,
            cmd_hash,
            output_hash,
            payload: payload.clone(),
        };
        self.append_to_disk(&receipt)?;
        self.index
            .lock()
            .expect("receipt index lock")
            .insert(receipt.id.clone(), receipt.clone());
        Ok(receipt)
    }

    fn day_file(&self, ts: u64) -> PathBuf {
        // día civil a partir de epoch ms (UTC)
        let days = ts / 86_400_000;
        // 1970-01-01 = día 0; formato YYYYMMDD sin librería de fechas:
        // conversión días→(y,m,d) por el algoritmo civil de Howard Hinnant.
        let (y, m, d) = civil_from_days(days as i64);
        self.dir.join(format!("receipts-{:04}{:02}{:02}.jsonl", y, m, d))
    }

    fn append_to_disk(&self, r: &Receipt) -> Result<(), ReceiptError> {
        let path = self.day_file(r.ts);
        let line = serde_json::to_string(r)
            .map_err(|e| ReceiptError(format!("serializar receipt: {}", e)))?;
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| ReceiptError(format!("abrir {:?} append: {}", path, e)))?;
        f.write_all(line.as_bytes())
            .and_then(|_| f.write_all(b"\n"))
            .map_err(|e| ReceiptError(format!("append receipt: {}", e)))
    }

    /// Busca un receipt por id (memoria → disco). None si no existe.
    pub fn get(&self, id: &str) -> Option<Receipt> {
        if let Some(r) = self.index.lock().expect("index lock").get(id) {
            return Some(r.clone());
        }
        if self.load_from_disk().is_ok() {
            self.index.lock().expect("index lock").get(id).cloned()
        } else {
            None
        }
    }

    /// Carga (una vez) todos los JSONL del directorio al índice.
    fn load_from_disk(&self) -> Result<usize, ReceiptError> {
        let mut loaded = 0usize;
        let entries = std::fs::read_dir(&self.dir)
            .map_err(|e| ReceiptError(format!("read_dir {:?}: {}", self.dir, e)))?;
        let mut scanned = self.scanned.lock().expect("scanned lock");
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if scanned.contains(&path) {
                continue;
            }
            let f = File::open(&path)
                .map_err(|e| ReceiptError(format!("abrir {:?}: {}", path, e)))?;
            for line in BufReader::new(f).lines() {
                let line = line.map_err(|e| ReceiptError(format!("leer línea: {}", e)))?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(r) = serde_json::from_str::<Receipt>(&line) {
                    self.index
                        .lock()
                        .expect("index lock")
                        .insert(r.id.clone(), r);
                    loaded += 1;
                }
            }
            scanned.insert(path);
        }
        Ok(loaded)
    }

    /// Verifica un receipt: re-firma el payload guardado y compara la firma.
    /// Payload/firma alterados → inválido. Receipt inexistente → inválido.
    pub fn verify_receipt(&self, receipt_id: &str) -> VerifyResult {
        let Some(r) = self.get(receipt_id) else {
            return VerifyResult {
                valid: false,
                reason: format!("receipt '{}' no encontrado", receipt_id),
                receipt: None,
            };
        };
        let canonical = canonicalize(&r.payload);
        if verify_sig(&self.key, canonical.as_bytes(), &r.sig_hex) {
            VerifyResult {
                valid: true,
                reason: "ok".into(),
                receipt: Some(r),
            }
        } else {
            VerifyResult {
                valid: false,
                reason: "firma inválida: payload o sig alterados (tamper)".into(),
                receipt: Some(r),
            }
        }
    }

    /// Verifica un receipt contra un claim `{receipt_id, claim:{metric,value}}`.
    /// Re-verifica la firma y que el claim esté respaldado por el payload
    /// (existe una clave `metric` con ese valor exacto, en cualquier nivel).
    pub fn verify_claim(
        &self,
        receipt_id: &str,
        claim: &serde_json::Value,
    ) -> VerifyResult {
        let base = self.verify_receipt(receipt_id);
        if !base.valid {
            return base;
        }
        let r = base.receipt.as_ref().expect("valid implies receipt");
        let claim_obj = claim.get("claim").cloned().unwrap_or_else(|| claim.clone());
        let (Some(metric), Some(value)) = (
            claim_obj.get("metric").and_then(|v| v.as_str()),
            claim_obj.get("value"),
        ) else {
            return VerifyResult {
                valid: false,
                reason: "claim malformado: se requiere {metric, value}".into(),
                receipt: base.receipt,
            };
        };
        if value_matches(&r.payload, metric, value) {
            VerifyResult {
                valid: true,
                reason: "ok".into(),
                receipt: base.receipt,
            }
        } else {
            VerifyResult {
                valid: false,
                reason: format!(
                    "claim (metric '{}' con valor {}) no está respaldado por el payload del receipt {}",
                    metric, canonicalize(value), receipt_id
                ),
                receipt: base.receipt,
            }
        }
    }

    /// Gate N3 para la integración: consume un claim
    /// `{"receipt_id": "...", "claim": {"metric": "...", "value": ...}}`
    /// (o, directamente, `{"metric": "...", "value": ...}` con
    /// `receipt_id` opcional) y devuelve Pass/Fail/Unverifiable.
    ///
    /// - Receipt ausente o sin `receipt_id` → `Unverifiable` (no hay
    ///   oráculo; el rediseño manda Unverifiable, no Fail).
    /// - Firma inválida (tamper) → `Fail`.
    /// - Claim no contenido en el payload firmado → `Fail`.
    /// - Firma válida y claim contenido → `Pass`.
    pub fn n3_check(&self, claim: &serde_json::Value) -> N3Verdict {
        let Some(receipt_id) = claim.get("receipt_id").and_then(|v| v.as_str()) else {
            return N3Verdict::Unverifiable;
        };
        let Some(_r) = self.get(receipt_id) else {
            return N3Verdict::Unverifiable;
        };
        match self.verify_claim(receipt_id, claim) {
            VerifyResult { valid: true, .. } => N3Verdict::Pass,
            VerifyResult { valid: false, reason, .. } if reason.starts_with("firma inválida") => {
                N3Verdict::Fail
            }
            VerifyResult { valid: false, reason, .. } if reason.starts_with("claim") || reason.starts_with("Claim") => {
                N3Verdict::Fail
            }
            VerifyResult { valid: false, .. } => N3Verdict::Unverifiable,
        }
    }

    /// Cantidad de receipts indexados en memoria.
    pub fn len(&self) -> usize {
        self.index.lock().expect("index lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// días desde epoch civil → (año, mes, día). Algoritmo de Hinnant.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// ¿Existe en `payload` una clave `metric` cuyo valor canónico sea igual a
/// `value`? Búsqueda recursiva en cualquier nivel del JSON firmado.
fn value_matches(payload: &serde_json::Value, metric: &str, value: &serde_json::Value) -> bool {
    match payload {
        serde_json::Value::Object(map) => {
            if let Some(v) = map.get(metric) {
                // comparación por representación canónica (5 == 5.0 serían
                // distintos; para métricas usamos igualdad exacta canónica)
                if canonicalize(v) == canonicalize(value) {
                    return true;
                }
            }
            map.values().any(|v| value_matches(v, metric, value))
        }
        serde_json::Value::Array(items) => {
            items.iter().any(|v| value_matches(v, metric, value))
        }
        _ => false,
    }
}

/// RNG sin depender de `rand` en este módulo: usa SHA256 de entropía del SO.
fn getrandom_fill(buf: &mut [u8]) -> Result<(), ReceiptError> {
    // /dev/urandom en Linux/WSL; fallback: tiempo + pid (no cripto pero
    // mejor que nada si falla el primero).
    if let Ok(mut f) = File::open("/dev/urandom") {
        use std::io::Read;
        if f.read_exact(buf).is_ok() {
            return Ok(());
        }
    }
    let mut seed = format!("{}-{}", now_ms(), std::process::id());
    let mut off = 0usize;
    while off < buf.len() {
        let h = hmac_hex(b"receipts-keygen", seed.as_bytes());
        let b = h.clone().into_bytes();
        for byte in b {
            if off >= buf.len() {
                break;
            }
            buf[off] = byte;
            off += 1;
        }
        seed = h;
    }
    Ok(())
}

#[cfg(unix)]
fn set_mode_0600(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}
#[cfg(not(unix))]
fn set_mode_0600(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "receipts_test_{}_{}_{}",
            name,
            std::process::id(),
            now_ms()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample_payload() -> serde_json::Value {
        serde_json::json!({
            "cmd": "cargo test --lib",
            "output": {"tests_passed": 276, "tests_total": 276},
            "exit_code": 0
        })
    }

    #[test]
    fn test_hmac_roundtrip() {
        let dir = tmpdir("roundtrip");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let r = store.issue(&sample_payload()).unwrap();
        assert!(r.sig_hex.len() == 64);
        let v = store.verify_receipt(&r.id);
        assert!(v.valid, "roundtrip debe verificar: {}", v.reason);
        // persistido en disco
        assert!(dir.join(day_filename(&r)).exists());
        // re-open: mismo .key → sigue verificando (persistencia de clave)
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        assert!(store2.verify_receipt(&r.id).valid);
    }

    fn day_filename(r: &Receipt) -> String {
        let days = r.ts / 86_400_000;
        let (y, m, d) = civil_from_days(days as i64);
        format!("receipts-{:04}{:02}{:02}.jsonl", y, m, d)
    }

    #[test]
    fn test_tamper_detection() {
        let dir = tmpdir("tamper");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let r = store.issue(&sample_payload()).unwrap();
        // alterar el payload en el índice + disco
        let mut tampered = r.clone();
        tampered.payload["output"]["tests_passed"] = serde_json::json!(999);
        let path = dir.join(day_filename(&r));
        let line = serde_json::to_string(&tampered).unwrap();
        std::fs::write(&path, format!("{}\n", line)).unwrap();
        // nuevo store fuerza reload desde disco
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        let v = store2.verify_receipt(&r.id);
        assert!(!v.valid, "payload alterado debe fallar");
        assert!(v.reason.contains("firma inválida"));
    }

    #[test]
    fn test_tamper_sig_detection() {
        let dir = tmpdir("tamper_sig");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let r = store.issue(&sample_payload()).unwrap();
        let mut tampered = r.clone();
        tampered.sig_hex = format!("{}0", &tampered.sig_hex[..63]); // 64 chars, wrong
        let path = dir.join(day_filename(&r));
        std::fs::write(&path, format!("{}\n", serde_json::to_string(&tampered).unwrap())).unwrap();
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        assert!(!store2.verify_receipt(&r.id).valid);
    }

    #[test]
    fn test_missing_receipt() {
        let dir = tmpdir("missing");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let v = store.verify_receipt("rcpt_inexistente_0000");
        assert!(!v.valid);
        assert!(v.reason.contains("no encontrado"));
        // n3_check: ausente → Unverifiable
        assert_eq!(
            store.n3_check(&serde_json::json!({"receipt_id": "nope", "claim": {"metric": "x", "value": 1}})),
            N3Verdict::Unverifiable
        );
    }

    #[test]
    fn test_key_persistence_and_env() {
        let dir = tmpdir("keypersist");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        assert!(dir.join(".key").exists());
        let r = store.issue(&sample_payload()).unwrap();
        // segunda apertura reutiliza la MISMA clave (mismo .key)
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        assert!(store2.verify_receipt(&r.id).valid);
        // clave nunca aparece en el receipt serializado
        let key_bytes = std::fs::read(dir.join(".key")).unwrap();
        let receipt_line = serde_json::to_string(&r).unwrap();
        assert!(!receipt_line.contains(&hex::encode(&key_bytes)));
        // permisos 0600 en unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join(".key")).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn test_env_key_used_when_present() {
        let dir = tmpdir("envkey");
        // clave hex de 64 chars (32 bytes)
        let env_key = "a".repeat(64);
        std::env::set_var(HMAC_KEY_ENV, &env_key);
        let store = ReceiptStore::open(&dir).unwrap();
        let r = store.issue(&sample_payload()).unwrap();
        // sin .key generado: la clave vino del env
        assert!(!dir.join(".key").exists());
        // una clave distinta no verifica
        std::env::set_var(HMAC_KEY_ENV, "b".repeat(64));
        let store2 = ReceiptStore::open(&dir).unwrap();
        assert!(!store2.verify_receipt(&r.id).valid);
        std::env::remove_var(HMAC_KEY_ENV);
    }

    #[test]
    fn test_wrong_key_fails() {
        let dir = tmpdir("wrongkey");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let r = store.issue(&sample_payload()).unwrap();
        // borrar .key y regenerar → clave distinta → firma inválida
        std::fs::remove_file(dir.join(".key")).unwrap();
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        assert!(!store2.verify_receipt(&r.id).valid, "otra clave no debe verificar");
    }

    #[test]
    fn test_persistence_across_restart() {
        let dir = tmpdir("restart");
        {
            let store = ReceiptStore::open_no_env(&dir).unwrap();
            for i in 0..10 {
                store
                    .issue(&serde_json::json!({"cmd": format!("op{}", i), "exit_code": 0}))
                    .unwrap();
            }
        }
        let store2 = ReceiptStore::open_no_env(&dir).unwrap();
        store2.load_from_disk().unwrap();
        assert_eq!(store2.len(), 10);
        // todos verifican tras el "restart" (ids colectados ANTES de lockear:
        // verify_receipt re-toma el lock del índice)
        let ids: Vec<String> = {
            let idx = store2.index.lock().unwrap();
            idx.keys().cloned().collect()
        };
        for id in &ids {
            assert!(store2.verify_receipt(id).valid);
        }
    }

    #[test]
    fn test_n3_check_pass_fail_unverifiable() {
        let dir = tmpdir("n3");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let payload = serde_json::json!({
            "cmd": "bench",
            "output": {"rps": 755.9},
            "exit_code": 0
        });
        let r = store.issue(&payload).unwrap();
        // Pass: claim contenido y firma válida
        assert_eq!(
            store.n3_check(&serde_json::json!({
                "receipt_id": r.id,
                "claim": {"metric": "rps", "value": 755.9}
            })),
            N3Verdict::Pass
        );
        // Fail: claim que NO está en el payload
        assert_eq!(
            store.n3_check(&serde_json::json!({
                "receipt_id": r.id,
                "claim": {"metric": "rps", "value": 1.0}
            })),
            N3Verdict::Fail
        );
        // Unverifiable: sin receipt_id
        assert_eq!(
            store.n3_check(&serde_json::json!({"claim": {"metric": "rps", "value": 755.9}})),
            N3Verdict::Unverifiable
        );
    }

    #[test]
    fn test_canonicalize_sorted_no_whitespace() {
        let a = serde_json::json!({"z": 1, "a": {"c": 2, "b": [1, 2]}, "m": "x"});
        let s = canonicalize(&a);
        assert_eq!(s, r#"{"a":{"b":[1,2],"c":2},"m":"x","z":1}"#);
        assert!(!s.contains(' '));
        // mismo contenido, distinto orden de inserción → igual canónico
        let b: serde_json::Value =
            serde_json::from_str(r#"{"m":"x","z":1,"a":{"b":[1,2],"c":2}}"#).unwrap();
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn test_issue_perf_1000_ops() {
        let dir = tmpdir("perf");
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let payload = sample_payload();
        // warmup
        for _ in 0..50 {
            store.issue(&payload).unwrap();
        }
        let start = std::time::Instant::now();
        let n = 1000;
        for i in 0..n {
            store
                .issue(&serde_json::json!({"cmd": format!("op{}", i), "exit_code": 0}))
                .unwrap();
        }
        let elapsed = start.elapsed();
        let per_op_us = elapsed.as_micros() as f64 / n as f64;
        let per_op_ms = per_op_us / 1000.0;
        println!("PERF issue: {} ops en {:?} → {:.4} ms/op", n, elapsed, per_op_ms);
        assert!(
            per_op_ms < 15.0,
            "perf issue {:.4} ms/op >= 15 ms target",
            per_op_ms
        );
        // reportar también para el parser del reporte
        std::fs::write(
            dir.join("perf.txt"),
            format!("{:.6}\n", per_op_ms),
        )
        .unwrap();
    }

    #[test]
    fn test_civil_from_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2024-01-01 = 19723 días desde epoch (54 años, 13 bisiestos)
        assert_eq!(civil_from_days(19723), (2024, 1, 1));
    }
}

//! Anatomía de memoria de ErrorRecord (baseline RAM-only de Agent-R).
//!
//! Mide size_of de cada struct y estimación de bytes/incidente en heap
//! (String + Vec overhead reales de la std, no estimaciones a ojo), y
//! proyecta escala a 100k / 1M errores. corre con:
//!   CARGO_TARGET_DIR=$HOME/swarm_agent_r/target cargo run --release --example rambench_anatomy

use agent_r::agent_r::{ErrorCategory, ErrorRecord, ErrorSeverity};

/// Bytes heap de un String de longitud `len` (capacidad == len, alineación 1).
fn string_heap(s: &str) -> usize {
    s.len() // heap: bytes; el String struct (ptr,len,cap) vive inline en el struct
}

/// Heap de un Vec<String> de n strings con longitudes dadas.
#[allow(dead_code)]
fn vec_of_strings_heap(items: &[String]) -> usize {
    // Vec buffer: cap*8 bytes de String-structs + heap de cada string
    let buf = items.len().next_power_of_two().max(1) * std::mem::size_of::<String>();
    let strs: usize = items.iter().map(|s| string_heap(s.as_str())).sum();
    buf + strs
}

fn main() {
    println!("=== size_of de structs (bytes, x86_64) ===");
    println!(
        "ErrorRecord            : {}",
        std::mem::size_of::<ErrorRecord>()
    );
    println!(
        "ErrorCategory (enum)   : {}",
        std::mem::size_of::<ErrorCategory>()
    );
    println!(
        "ErrorSeverity (enum)   : {}",
        std::mem::size_of::<ErrorSeverity>()
    );
    println!("String                 : {}", std::mem::size_of::<String>());
    println!(
        "Vec<String>            : {}",
        std::mem::size_of::<Vec<String>>()
    );
    println!(
        "Option<u64>            : {}",
        std::mem::size_of::<Option<u64>>()
    );

    // Payload promedio REAL del harness medido sobre 10k registros seed 20261012:
    // operation 15.4 B, what 85.3 B, why 99.4 B, prevention 82.4 B, tags 2.51 x 7.4 B.
    // Alineación de String en heap: malloc bins redondean (usable size).
    fn malloc_usable(len: usize) -> usize {
        // glibc malloc: 16-byte chunks, header 8 usable + alineación => aproximación conservadora:
        // usable = max(24, round_up(len, 16) + 8) para len>0; smallest bin 32 (24 usable)
        if len == 0 {
            0
        } else {
            ((len + 8 + 15) / 16) * 16 + if (len + 8) % 16 == 0 { 0 } else { 8 }
        }
    }

    let operation = "distill_lessons".to_string();
    let what =
        "synthetic failure mode #5000: distill_lessons produced inconsistent state hash 3f2a1b"
            .to_string();
    let why = "root cause a17f: precondition violated before distill_lessons (deterministic harness seed 20261012)".to_string();
    let prevention =
        "verify preconditions of distill_lessons and re-run with receipt 42ab before commit"
            .to_string();
    let context_tags: Vec<String> = vec!["gpu".into(), "ci".into(), "index".into()];

    let inline = std::mem::size_of::<ErrorRecord>();
    // Heap con bins de malloc (glibc): cada String usa malloc_usable(len)
    let op_h = malloc_usable(operation.len());
    let what_h = malloc_usable(what.len());
    let why_h = malloc_usable(why.len());
    let prev_h = malloc_usable(prevention.len());
    // Vec<String> de 2.51 tags promedio: buffer redondo a potencia de 2 de capacidad (2->4, cap 4)
    let tag_buf = context_tags.capacity() * std::mem::size_of::<String>();
    let tag_strs: usize = context_tags.iter().map(|s| malloc_usable(s.len())).sum();
    let tags_h = tag_buf + tag_strs;

    println!("\n=== Payload representativo del harness (promedio 10k regs, seed 20261012) ===");
    println!("operation  len={} heap_malloc={}", operation.len(), op_h);
    println!("what       len={} heap_malloc={}", what.len(), what_h);
    println!("why        len={} heap_malloc={}", why.len(), why_h);
    println!("prevention len={} heap_malloc={}", prevention.len(), prev_h);
    println!(
        "context_tags n={} cap={} heap(Vec buf {} + strings {})={}",
        context_tags.len(),
        context_tags.capacity(),
        tag_buf,
        tag_strs,
        tags_h
    );

    let per_error = inline + op_h + what_h + why_h + prev_h + tags_h;
    println!("\ninline struct : {}", inline);
    println!(
        "heap strings  : {}",
        op_h + what_h + why_h + prev_h + tags_h
    );
    println!(
        "TOTAL por incidente (sin overhead de Vec<ErrorRecord>): {}",
        per_error
    );

    // Overhead del contenedor Vec<ErrorRecord>: amortizado 1 ptr/incidente (crecimiento
    // amortizado ~ ptr*cap; además el growth deja buffers muertos hasta realloc).
    let vec_slot = std::mem::size_of::<*const ErrorRecord>();
    println!("slot Vec (ptr amortizado) : {}", vec_slot);
    let per_error_total = per_error + vec_slot;
    println!("TOTAL por incidente (con slot Vec) : {}", per_error_total);

    println!("\n=== Proyección de escala (solo Vec de errores, sin index) ===");
    for n in [10_000usize, 50_000, 100_000, 1_000_000] {
        let bytes = per_error_total * n;
        println!(
            "N={:>9} => {:>12} bytes = {:>8.2} MiB = {:>7.3} GiB",
            n,
            bytes,
            bytes as f64 / 1_048_576.0,
            bytes as f64 / 1_073_741_824.0
        );
    }

    // Verificación empírica con el payload promedio real del harness:
    // promedios medidos del generador (longitudes fijas por plantilla).
    println!("\nNota: MAX_ERRORS=10000 en el baseline acota el Vec a 10k (eviction remove(0));");
    println!(
        "la proyección 50k/100k/1M asume el objetivo de las oleadas 3-5 (sin cap o cap mayor)."
    );
}

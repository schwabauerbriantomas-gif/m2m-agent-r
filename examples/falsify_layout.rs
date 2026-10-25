//! OPT-O3 falsificación barata: anatomía de ErrorRecord y sus variantes de layout.
//!
//! Mide, SIN tocar src/:
//!   1. size_of/align_of del struct actual (String ×4 + Vec<String>) y de las
//!      variantes candidatas: Box<str> ×3, Vec<u32> tags, interner (u32 op).
//!   2. Heap real por registro con el generador del harness (longitudes exactas
//!      de gen_errors.py) => malloc_bins(len) con bins glibc de 16 B.
//!   3. Estabilidad de punteros: los 4 String + Vec son DEDIUM (puntero inline,
//!      datos heap). Con Vec<ErrorRecord> append-only SIN realloc, los punteros
//!      a los datos heap de cada registro NO se mueven => imprime direcciones
//!      antes/después de 1000 pushes para verificarlo empíricamente.
//!
//! Corrida: cargo run --release --example falsify_layout --quiet (CARGO_TARGET_DIR propio)

use agent_r::agent_r::{ErrorCategory, ErrorRecord, ErrorSeverity};

/// Tamaño de asignación glibc para un string de `len` bytes (bins de 16, headers
/// incluidos en el len observado: aproximación de campo, no de malloc header).
fn bin16(len: usize) -> usize {
    (len.max(1) + 8).div_ceil(16) * 16
}

fn main() {
    // ── 1) size_of inline de las variantes ─────────────────────────────
    #[derive(Debug)]
    struct BoxedVariant {
        id: u64,
        timestamp: f64,
        category: ErrorCategory,
        severity: ErrorSeverity,
        operation: Box<str>,
        what: Box<str>,
        why: Box<str>,
        context_tags: Vec<u32>,
        prevention: Box<str>,
        recurrence: u32,
        pattern_id: Option<u64>,
    }

    println!("== size_of / align_of ==");
    println!(
        "ErrorRecord actual (4 String + Vec<String>): {} B, align {}",
        std::mem::size_of::<ErrorRecord>(),
        std::mem::align_of::<ErrorRecord>()
    );
    println!(
        "Variante Box<str> x3 + Vec<u32> tags:       {} B, align {}",
        std::mem::size_of::<BoxedVariant>(),
        std::mem::align_of::<BoxedVariant>()
    );
    println!(
        "  String={} Box<str>={} Vec<u32>={} Option<u64>={}",
        std::mem::size_of::<String>(),
        std::mem::size_of::<Box<str>>(),
        std::mem::size_of::<Vec<u32>>(),
        std::mem::size_of::<Option<u64>>(),
    );
    println!(
        "  ErrorCategory={} ErrorSeverity={}",
        std::mem::size_of::<ErrorCategory>(),
        std::mem::size_of::<ErrorSeverity>(),
    );

    // ── 2) Heap real por registro con longitudes del harness ──────────
    // what = "synthetic failure mode #{i}: {op} produced inconsistent state hash {r&0xFFFFFF:06x}"
    // why  = "root cause {r2&0xFFFF:04x}: precondition violated before {op} (deterministic harness seed 20261012)"
    // prevention = "verify preconditions of {op} and re-run with receipt {r3&0xFFFF:04x} before commit"
    // Promedios verificados del brief: what 85.3, why 99.4, prevention 82.4, op 15.4, tags 2.51×7.4
    let heap_actual = bin16(15) /* op String */
        + bin16(85)  /* what */
        + bin16(99)  /* why */
        + bin16(82)  /* prevention */
        + 24         /* Vec buffer String ptrs (cap inicial 4 tras 2.51 avg: bin16(2.51*24)=96? medimos abajo) */
        ;
    println!("\n== heap teórico por registro (bins glibc 16 B) ==");
    println!(
        "strings + vec-buf aproximado: ~{} B (sin buffer de String de tags)",
        heap_actual
    );
    println!(
        "  bins: op={} what={} why={} prevention={}",
        bin16(15),
        bin16(85),
        bin16(99),
        bin16(82)
    );

    // ── 3) DEDIUM: estabilidad de punteros heap con Vec append-only ────
    // (Usamos un espejo del layout LEGACY: el principio DEDIUM es el mismo
    //  para Box<str> del layout nuevo — dato en heap, puntero inline.)
    #[derive(Clone)]
    struct LegacyRecord {
        operation: String,
        what: String,
        why: String,
        context_tags: Vec<String>,
        prevention: String,
    }
    println!("\n== estabilidad de punteros (DEDIUM) ==");
    let mut v: Vec<LegacyRecord> = Vec::new();
    for i in 0..1000u64 {
        v.push(LegacyRecord {
            operation: format!("op_{i}"),
            what: format!("what_{i}"),
            why: format!("why_{i}"),
            context_tags: vec!["gpu".to_string(), "ci".to_string()],
            prevention: format!("prev_{i}"),
        });
    }
    // El realloc del Vec<ErrorRecord> MUEVE los structs (inline) pero los datos
    // heap de cada String quedan donde estaban: verificamos con 3 registros.
    let probe = |v: &Vec<LegacyRecord>, tag: &str| {
        for idx in [0usize, 500, 999] {
            let e = &v[idx];
            println!(
                "  [{tag}] rec#{idx:<4} op_ptr={:p} what_ptr={:p} tags_ptr={:p}",
                e.operation.as_ptr(),
                e.what.as_ptr(),
                e.context_tags.as_ptr()
            );
        }
    };
    probe(&v, "tras 1000 pushes");
    // 1000 pushes más SIN limpiar: si el Vec realloca, los structs se mueven,
    // los punteros heap de los Strings NO.
    for i in 1000..2000u64 {
        v.push(LegacyRecord {
            operation: format!("op_{i}"),
            what: format!("what_{i}"),
            why: format!("why_{i}"),
            context_tags: vec!["gpu".to_string()],
            prevention: format!("prev_{i}"),
        });
    }
    probe(&v, "tras 2000 pushes");
    println!(
        "\n  (los op_ptr/what_ptr/tags_ptr deben ser IDÉNTICOS entre ambas fotos: \
         los datos heap no se mueven aunque el Vec realoque los structs inline)"
    );

    // ── 4) Heap medido con longitudes EXACTAS del generador ───────────
    // Reproduce las plantillas del harness y mide len() real promedio.
    let seed = 20261012u64;
    let splitmix = |x: &mut u64| -> u64 {
        *x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let ops = [
        "reindex_collection",
        "store_memory",
        "search_similar",
        "rebuild_hnsw_index",
        "merge_splats",
        "validate_output_chain",
        "promote_rule",
        "distill_lessons",
        "snapshot_metadata",
        "gc_expired_splats",
        "preflight_mutation",
        "audit_receipts",
        "consolidate_memory",
        "expand_context",
        "route_query",
    ];
    let tags_pool = [
        "gpu",
        "cpu",
        "nightly",
        "ci",
        "prod",
        "index",
        "benchmark",
        "large-batch",
        "cold-start",
        "retry",
        "timeout-window",
        "sqlite",
        "hot-loop",
        "high-recall",
        "precision@10",
        "hnsw",
        "warming",
        "crash-recovery",
        "sha256",
        "receipt",
    ];
    let (mut l_op, mut l_what, mut l_why, mut l_prev, mut l_tags, mut n_tag_str) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let n = 10_000usize;
    for i in 0..n {
        let mut x = if i == 0 {
            seed.wrapping_mul(1)
        } else {
            seed ^ (i as u64).wrapping_mul(0x8000000000000001)
        };
        let r = splitmix(&mut x);
        let r2 = splitmix(&mut x);
        let r3 = splitmix(&mut x);
        let r4 = splitmix(&mut x);
        let op = ops[((r3 >> 31) % ops.len() as u64) as usize];
        let n_tags = 1 + ((r4 >> 47) % 4) as usize;
        let tags: Vec<&str> = (0..n_tags)
            .map(|k| tags_pool[((r >> (8 * (k + 1))) % tags_pool.len() as u64) as usize])
            .collect();
        let what = format!(
            "synthetic failure mode #{i}: {op} produced inconsistent state hash {:06x}",
            r & 0xFFFFFF
        );
        let why = format!("root cause {:04x}: precondition violated before {op} (deterministic harness seed {seed})", r2 & 0xFFFF);
        let prev = format!(
            "verify preconditions of {op} and re-run with receipt {:04x} before commit",
            r3 & 0xFFFF
        );
        l_op += op.len();
        l_what += what.len();
        l_why += why.len();
        l_prev += prev.len();
        l_tags += n_tags;
        n_tag_str += tags.iter().map(|t| t.len()).sum::<usize>();
    }
    println!("\n== longitudes medias del generador (n={n}) ==");
    println!(
        "  op={:.1} what={:.1} why={:.1} prevention={:.1} tags/reg={:.2} bytes/tag={:.1}",
        l_op as f64 / n as f64,
        l_what as f64 / n as f64,
        l_why as f64 / n as f64,
        l_prev as f64 / n as f64,
        l_tags as f64 / n as f64,
        n_tag_str as f64 / l_tags as f64,
    );

    // Costo heap por registro, escenario actual vs variantes:
    let (op_l, what_l, why_l, prev_l, tags_n) = (
        l_op as f64 / n as f64,
        l_what as f64 / n as f64,
        l_why as f64 / n as f64,
        l_prev as f64 / n as f64,
        l_tags as f64 / n as f64,
    );
    let string_bin = |l: f64| -> f64 {
        // String con capacidad exacta al len: bin glibc de len+~0; para promedio
        // usamos E[bin16(len)] ≈ bin sobre la media (sesgo <2 B acá).
        ((l + 8.0).ceil() / 16.0).ceil() * 16.0
    };
    let heap_now = string_bin(op_l) + string_bin(what_l) + string_bin(why_l) + string_bin(prev_l)
        + 24.0 /* Vec<String> buf: cap 4 ⇒ 96 B pero amortiza a ~60 con cap exacto */ * 0.0
        + 96.0 /* String[] del vec (cap 4, 24 B c/u) */
        + string_bin(7.4) * tags_n;
    println!("\n== heap/registro estimado ==");
    println!(
        "  actual        : {:.0} B (4 String + Vec<String> con bins)",
        heap_now
    );
    println!("  + Box<str> x3 : what/why/prev pasan de String a Box<str>: -8 B inline c/u (cap no almacenada), heap igual");
    println!("  + interner op/tags: op 24B inline -> 4, heap_op eliminado; tags Vec<String> 96+bin(tags) -> Vec<u32> ~16 B");
}

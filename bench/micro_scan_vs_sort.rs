//! Micro-bench: qué domina recall_errors — ¿scan+score o el sort?
//! Reproduce EXACTAMENTE la lógica de src/agent_r.rs:220-261 sobre datos
//! sintéticos con la MISMA distribución que scripts/rambench/gen_errors.py
//! (15 operaciones, 20 tags, severities warning/error/critical).
//! Falsifica/imposta: si sort << scan, la optimización top-k no sirve.
#![allow(dead_code)]

#[derive(Clone, Debug, PartialEq)]
enum Severity { Critical, Error, Warning }

struct Rec {
    operation: String,
    tags: Vec<String>,
    recurrence: u32,
    timestamp: f64,
    severity: Severity,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = ((z ^ (z >> 30)) as u128 * 0xBF58476D1CE4E5B9 as u128) as u64;
    z = ((z ^ (z >> 27)) as u128 * 0x94D049BB133111EB as u128) as u64;
    z ^ (z >> 31)
}

const OPS: [&str; 15] = ["reindex_collection","store_memory","search_similar","rebuild_hnsw_index","merge_splats","validate_output_chain","promote_rule","distill_lessons","snapshot_metadata","gc_expired_splats","preflight_mutation","audit_receipts","consolidate_memory","expand_context","route_query"];
const TAGS: [&str; 20] = ["gpu","cpu","nightly","ci","prod","index","benchmark","large-batch","cold-start","retry","timeout-window","sqlite","hot-loop","high-recall","precision@10","hnsw","warming","crash-recovery","sha256","receipt"];

fn gen(n: usize, seed: u64) -> Vec<Rec> {
    let mut recs = Vec::with_capacity(n);
    for i in 0..n {
        let r = splitmix64(seed ^ if i > 0 { (i as u64).wrapping_mul(0x8000000000000001) } else { 1 });
        let r2 = splitmix64(r); let r3 = splitmix64(r2); let r4 = splitmix64(r3);
        let op = OPS[((r3 >> 31) % 15) as usize];
        let sev = match (r2 >> 17) % 3 { 0 => Severity::Critical, 1 => Severity::Error, _ => Severity::Warning };
        let ntags = 1 + ((r4 >> 47) % 4) as usize;
        let tags = (0..ntags).map(|k| TAGS[((r >> (8*(k+1))) % 20) as usize].to_string()).collect::<Vec<_>>();
        recs.push(Rec { operation: op.to_string(), tags, recurrence: (r % 4) as u32, timestamp: i as f64, severity: sev });
    }
    recs
}

#[inline]
fn score(e: &Rec, op: &str, tags: &[&str], last_ts: f64) -> f64 {
    let mut s = 0.0;
    if e.operation == op { s += 10.0; }
    else if e.operation.contains(op) || op.contains(&e.operation) { s += 5.0; }
    for t in tags { if e.tags.iter().any(|x| x == t) { s += 3.0; } }
    let age_h = (last_ts - e.timestamp) / 3600.0;
    s += (24.0 - age_h.min(24.0)) * 0.5;
    s += e.recurrence as f64 * 2.0;
    s += match e.severity { Severity::Critical => 5.0, Severity::Error => 3.0, Severity::Warning => 1.0 };
    s
}

/// Selección top-k en O(n) promedio vía quickselect de std.
/// Después de la llamada, los primeros k elementos son los k mayores
/// (desordenados entre sí) según el comparador descendente.
fn select_topk_desc(v: &mut [(f64, &Rec)], k: usize) {
    if k == 0 || v.len() <= k { return; }
    v.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let qop = "distill_lessons"; let qtags: [&str; 2] = ["gpu", "ci"];
    let recs = gen(n, 20261012);
    let last_ts = recs.last().unwrap().timestamp;

    // checksum para que el compilador no elida
    let mut sink: f64 = 0.0;

    // (1) SOLO scan+score (construcción del Vec, sin sort)
    let t0 = std::time::Instant::now();
    let scored: Vec<(f64, &Rec)> = recs.iter().filter_map(|e| {
        let s = score(e, qop, &qtags, last_ts);
        if s > 0.0 { Some((s, e)) } else { None }
    }).collect();
    let t_scan = t0.elapsed();
    sink += scored.len() as f64;

    // (2) sort completo O(n log n) — como el código actual
    let t0 = std::time::Instant::now();
    let mut sorted = scored.clone();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top20_fullsort: Vec<&Rec> = sorted.into_iter().take(20).map(|(_, e)| e).collect();
    let t_sort = t0.elapsed();
    sink += top20_fullsort.len() as f64;

    // (3) selección top-20 lineal (partición de quickselect estilo pdqselect)
    let t0 = std::time::Instant::now();
    let mut v = scored.clone();
    select_topk_desc(&mut v, 20);
    let top20_sel: Vec<&Rec> = v.into_iter().take(20).map(|(_, e)| e).collect();
    let t_select = t0.elapsed();
    sink += top20_sel.len() as f64;

    // (4) BinaryHeap max de tamaño 20 (scan + heap) — score como bits (total order)
    let t0 = std::time::Instant::now();
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(21);
    for (i, e) in recs.iter().enumerate() {
        let s = score(e, qop, &qtags, last_ts);
        if s <= 0.0 { continue; }
        if heap.len() < 20 { heap.push(Reverse((s.to_bits(), i))); }
        else if s.to_bits() > heap.peek().unwrap().0 .0 { heap.pop(); heap.push(Reverse((s.to_bits(), i))); }
    }
    let t_heap = t0.elapsed();
    sink += heap.len() as f64;

    // (5) falso-positivo check: ¿el top-20 de selección == top-20 de sort?
    let ids_sorted: Vec<usize> = { let mut x: Vec<usize> = top20_fullsort.iter().map(|r| r.operation.clone().len() + r.tags.len()).collect(); x.sort(); x };
    let ids_sel: Vec<usize> = { let mut x: Vec<usize> = top20_sel.iter().map(|r| r.operation.clone().len() + r.tags.len()).collect(); x.sort(); x };
    let eq = ids_sorted == ids_sel;

    let ns = |d: std::time::Duration| d.as_nanos() as f64 / 1e6;
    println!("n={} scored_len={} (score>0 SIEMPRE: severity>=1)", n, scored.len());
    println!("scan+score_vec : {:.3} ms", ns(t_scan));
    println!("sort_completo  : {:.3} ms  (scan+sort = {:.3})", ns(t_sort), ns(t_scan + t_sort));
    println!("select_top20   : {:.3} ms", ns(t_select));
    println!("heap_top20     : {:.3} ms", ns(t_heap));
    println!("select==sort_top20: {}", eq);
    println!("sink={:.0}", sink);
}

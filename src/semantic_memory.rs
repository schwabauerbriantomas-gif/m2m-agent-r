//! SemanticMemoryDB — Semantic Memory for AI Agents.
//! Simplified port. See Python version for full features.

use std::collections::HashMap;

use crate::bm25_index::BM25Index;
use crate::embedding_model::Encoder;
use crate::splats::SplatStore;
use crate::config::M2MConfig;
use ndarray::Array1;
use serde::Serialize;

/// Category keywords for auto-categorization.
const CATEGORY_KEYWORDS: &[(&str, &[&str])] = &[
    ("decision", &["decided", "decision", "chose", "elected", "agreed", "plan"]),
    ("preference", &["prefers", "likes", "loves", "favorite", "default"]),
    ("project", &["project", "implement", "build", "feature", "sprint"]),
    ("error", &["error", "bug", "failed", "crash", "exception", "broken"]),
    ("learning", &["learned", "lesson", "discovered", "realized", "insight"]),
    ("task", &["todo", "task", "pending", "reminder", "need to"]),
];

/// Infer a category from text using keyword matching.
pub fn auto_categorize(text: &str) -> Option<&'static str> {
    let text_lower = text.to_lowercase();
    let mut best_cat = None;
    let mut best_count = 0;

    for (cat, keywords) in CATEGORY_KEYWORDS {
        let count = keywords.iter().filter(|kw| text_lower.contains(*kw)).count();
        if count > best_count {
            best_count = count;
            best_cat = Some(cat);
        }
    }

    if best_count > 0 { best_cat.map(|v| *v) } else { None }
}

/// A single memory search result.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryResult {
    pub id: String,
    pub document: Option<String>,
    pub metadata: serde_json::Value,
    pub score: f64,
    pub vector_score: f64,
    pub bm25_score: f64,
}

/// Fusion method for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    Rrf,
    Weighted,
    VectorOnly,
    Bm25Only,
}

/// Semantic Memory Database for AI Agents.
pub struct SemanticMemoryDB {
    encoder: Option<Box<dyn Encoder>>,
    latent_dim: usize,
    hybrid_weight: f64,
    fusion_method: FusionMethod,
    do_auto_categorize: bool,
    store: SplatStore,
    bm25: BM25Index,
    documents: HashMap<String, String>,
    doc_metadata: HashMap<String, serde_json::Value>,
    timestamps: HashMap<String, f64>,
    deleted: std::collections::HashSet<String>,
    query_count: u64,
    add_count: u64,
}

impl SemanticMemoryDB {
    pub fn new(latent_dim: usize) -> Self {
        let mut config = M2MConfig::default();
        config.latent_dim = latent_dim;

        Self {
            encoder: None,
            latent_dim,
            hybrid_weight: 0.6,
            fusion_method: FusionMethod::Rrf,
            do_auto_categorize: false,
            store: SplatStore::new(config),
            bm25: BM25Index::new(),
            documents: HashMap::new(),
            doc_metadata: HashMap::new(),
            timestamps: HashMap::new(),
            deleted: std::collections::HashSet::new(),
            query_count: 0,
            add_count: 0,
        }
    }

    pub fn with_encoder(mut self, encoder: Box<dyn Encoder>) -> Self {
        self.latent_dim = encoder.dim();
        self.encoder = Some(encoder);
        self
    }

    pub fn with_auto_categorize(mut self) -> Self {
        self.do_auto_categorize = true;
        self
    }

    fn now_secs() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    fn encode_text(&self, text: &str) -> Result<Array1<f32>, String> {
        match &self.encoder {
            Some(enc) => enc.encode_one(text).map_err(|e| e.to_string()),
            None => Err("No encoder set".into()),
        }
    }

    /// Store a memory with automatic embedding.
    pub fn store(&mut self, text: &str, metadata: Option<serde_json::Value>) -> Result<String, String> {
        if text.trim().is_empty() {
            return Err("text must be non-empty".into());
        }

        let id = format!("{:016x}", rand::random::<u64>());

        let embedding = self.encode_text(text)?;
        self.store_with_vector(text, &embedding, metadata, Some(&id))
    }

    /// Store a memory with a pre-computed vector.
    pub fn store_with_vector(
        &mut self,
        text: &str,
        vector: &Array1<f32>,
        metadata: Option<serde_json::Value>,
        doc_id: Option<&str>,
    ) -> Result<String, String> {
        if text.trim().is_empty() {
            return Err("text must be non-empty".into());
        }

        let id = doc_id.unwrap_or(&format!("{:016x}", rand::random::<u64>())).to_string();

        let vec2d = vector.view().insert_axis(ndarray::Axis(0)).to_owned();
        self.store.add_splat(&vec2d);

        self.documents.insert(id.clone(), text.to_string());
        self.doc_metadata.insert(id.clone(), metadata.unwrap_or(serde_json::Value::Null));
        self.bm25.add(&id, text);
        self.timestamps.insert(id.clone(), Self::now_secs());
        self.deleted.remove(&id);
        self.add_count += 1;

        Ok(id)
    }

    /// Search memories using hybrid search.
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<MemoryResult>, String> {
        if query.trim().is_empty() {
            return Err("query must not be empty".into());
        }

        self.query_count += 1;

        // Vector search
        let mut vector_score_map: HashMap<String, f64> = HashMap::new();
        if self.fusion_method != FusionMethod::Bm25Only {
            let query_vec = self.encode_text(query)?;
            let results = self.store.find_neighbors(&query_vec.view(), k * 3);
            for (i, r) in results.iter().enumerate() {
                let doc_id = format!("vec_{}", r.index);
                if self.deleted.contains(&doc_id) { continue; }
                let score = 1.0 - r.distance;
                vector_score_map.insert(doc_id.clone(), score.into());
            }
        }

        // BM25 search
        let mut bm25_score_map: HashMap<String, f64> = HashMap::new();
        if self.fusion_method != FusionMethod::VectorOnly {
            let bm25_results = self.bm25.search(query, k * 3);
            for (doc_id, score) in &bm25_results {
                if self.deleted.contains(doc_id) { continue; }
                bm25_score_map.insert(doc_id.clone(), *score);
            }
        }

        // Fusion
        let mut fused_scores: HashMap<String, f64> = HashMap::new();
        let all_ids: std::collections::HashSet<&String> =
            vector_score_map.keys().chain(bm25_score_map.keys()).collect();

        match self.fusion_method {
            FusionMethod::VectorOnly => {
                fused_scores = vector_score_map.clone();
            }
            FusionMethod::Bm25Only => {
                fused_scores = bm25_score_map.clone();
            }
            FusionMethod::Rrf => {
                let rrf_k = 60.0;
                let mut v_ranked: Vec<_> = vector_score_map.iter().collect();
                v_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let v_rank_map: HashMap<&String, usize> = v_ranked.iter().enumerate().map(|(r, (id, _))| (*id, r)).collect();

                let mut b_ranked: Vec<_> = bm25_score_map.iter().collect();
                b_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let b_rank_map: HashMap<&String, usize> = b_ranked.iter().enumerate().map(|(r, (id, _))| (*id, r)).collect();

                for id in &all_ids {
                    let v_rank = v_rank_map.get(id).copied().unwrap_or(v_ranked.len());
                    let b_rank = b_rank_map.get(id).copied().unwrap_or(b_ranked.len());
                    let v_rrf = self.hybrid_weight / (rrf_k + v_rank as f64 + 1.0);
                    let b_rrf = (1.0 - self.hybrid_weight) / (rrf_k + b_rank as f64 + 1.0);
                    fused_scores.insert((*id).clone(), v_rrf + b_rrf);
                }
            }
            FusionMethod::Weighted => {
                for id in &all_ids {
                    let v = vector_score_map.get(*id).copied().unwrap_or(0.0);
                    let b = bm25_score_map.get(*id).copied().unwrap_or(0.0);
                    fused_scores.insert((*id).to_string(), self.hybrid_weight * v + (1.0 - self.hybrid_weight) * b);
                }
            }
        }

        // Build results
        let mut ranked: Vec<_> = fused_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        let results: Vec<MemoryResult> = ranked
            .into_iter()
            .map(|(id, fused_score)| {
                let v_score = vector_score_map.get(&id).copied().unwrap_or(0.0);
                let b_score = bm25_score_map.get(&id).copied().unwrap_or(0.0);

                let doc_text = self.documents.get(&id).cloned()
                    .or_else(|| self.bm25.get_doc(&id).map(|s| s.to_string()));

                let meta = self.doc_metadata.get(&id).cloned().unwrap_or(serde_json::Value::Null);

                MemoryResult {
                    id,
                    document: doc_text,
                    metadata: meta,
                    score: fused_score,
                    vector_score: v_score,
                    bm25_score: b_score,
                }
            })
            .collect();

        Ok(results)
    }

    /// Delete a memory by ID.
    pub fn delete(&mut self, id: &str) -> bool {
        if self.documents.contains_key(id) {
            self.deleted.insert(id.to_string());
            self.bm25.remove(id);
            self.timestamps.remove(id);
            self.doc_metadata.remove(id);
            self.documents.remove(id);
            true
        } else {
            false
        }
    }

    /// Get a specific memory by ID.
    pub fn get(&self, id: &str) -> Option<serde_json::Value> {
        if self.deleted.contains(id) {
            return None;
        }
        self.documents.get(id).map(|doc| {
            serde_json::json!({
                "id": id,
                "document": doc,
                "metadata": self.doc_metadata.get(id).unwrap_or(&serde_json::Value::Null),
            })
        })
    }

    /// Get memory statistics.
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_memories": self.documents.len() - self.deleted.len(),
            "total_queries": self.query_count,
            "total_adds": self.add_count,
            "bm25_indexed": self.bm25.n_docs(),
            "hybrid_weight": self.hybrid_weight,
            "auto_categorize": self.do_auto_categorize,
        })
    }

    /// Clear all memories.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.doc_metadata.clear();
        self.timestamps.clear();
        self.deleted.clear();
        self.bm25.clear();
        let mut config = M2MConfig::default();
        config.latent_dim = self.latent_dim;
        self.store = SplatStore::new(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding_model::HashEncoder;

    fn make_test_db() -> SemanticMemoryDB {
        SemanticMemoryDB::new(64)
            .with_encoder(Box::new(HashEncoder::new(64)))
            .with_auto_categorize()
    }

    #[test]
    fn test_store_and_search() {
        let mut db = make_test_db();
        let _id = db.store("The team decided to use Rust for the project", None).unwrap();
        let results = db.search("Rust project decision", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_delete() {
        let mut db = make_test_db();
        let id = db.store("some memory", None).unwrap();
        assert!(db.delete(&id));
        assert!(db.get(&id).is_none());
    }

    #[test]
    fn test_auto_categorize() {
        assert_eq!(auto_categorize("We decided to use M2M"), Some("decision"));
        assert_eq!(auto_categorize("There is a bug in the code"), Some("error"));
        assert!(auto_categorize("random text xyz").is_none());
    }

    #[test]
    fn test_stats() {
        let db = make_test_db();
        let stats = db.stats();
        assert_eq!(stats["total_memories"], 0);
        assert_eq!(stats["auto_categorize"], true);
    }

    #[test]
    fn test_empty_query_error() {
        let mut db = make_test_db();
        assert!(db.search("", 5).is_err());
    }
}

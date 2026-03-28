//! Entity extractor for M2M.
//! Ported from m2m-vector-search Python.

use ndarray::Array1;
use regex::Regex;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct EntityCandidate {
    pub text: String,
    pub entity_type: String,
    pub score: f64,
    pub count: usize,
    pub embedding: Option<Array1<f32>>,
    pub cluster_id: i32,
    pub start_positions: Vec<usize>,
}

pub struct M2MEntityExtractor {
    email_re: Regex,
    url_re: Regex,
    phone_re: Regex,
    date_re: Regex,
    money_re: Regex,
    stopwords: HashSet<String>,
}

impl M2MEntityExtractor {
    pub fn new() -> Self {
        Self {
            email_re: Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap(),
            url_re: Regex::new(r"https?://[^\s]+").unwrap(),
            phone_re: Regex::new(r"\d{3}[-.]?\d{3}[-.]?\d{4}").unwrap(),
            date_re: Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap(),
            money_re: Regex::new(r"\$\d+(?:,\d{3})*(?:\.\d{2})?").unwrap(),
            stopwords: ["el","la","los","las","un","una","de","del","a","con","en","para","por","y","o","que","es","son","the","an","and","or","in","on","at","to","for","with","is","are"].iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn extract(&self, text: &str) -> Vec<EntityCandidate> {
        let mut candidates: HashMap<String, EntityCandidate> = HashMap::new();

        for m in self.email_re.find_iter(text) {
            candidates.insert(m.as_str().to_lowercase(), EntityCandidate {
                text: m.as_str().to_string(),
                entity_type: "contact".into(),
                score: 1.0,
                count: 1,
                embedding: None,
                cluster_id: -1,
                start_positions: vec![m.start()],
            });
        }

        for m in self.url_re.find_iter(text) {
            candidates.insert(m.as_str().to_lowercase(), EntityCandidate {
                text: m.as_str().to_string(),
                entity_type: "url".into(),
                score: 1.0,
                count: 1,
                embedding: None,
                cluster_id: -1,
                start_positions: vec![m.start()],
            });
        }

        for m in self.phone_re.find_iter(text) {
            candidates.insert(m.as_str().to_lowercase(), EntityCandidate {
                text: m.as_str().to_string(),
                entity_type: "contact".into(),
                score: 1.0,
                count: 1,
                embedding: None,
                cluster_id: -1,
                start_positions: vec![m.start()],
            });
        }

        for m in self.date_re.find_iter(text) {
            candidates.insert(m.as_str().to_lowercase(), EntityCandidate {
                text: m.as_str().to_string(),
                entity_type: "date".into(),
                score: 1.0,
                count: 1,
                embedding: None,
                cluster_id: -1,
                start_positions: vec![m.start()],
            });
        }

        for m in self.money_re.find_iter(text) {
            candidates.insert(m.as_str().to_lowercase(), EntityCandidate {
                text: m.as_str().to_string(),
                entity_type: "money".into(),
                score: 1.0,
                count: 1,
                embedding: None,
                cluster_id: -1,
                start_positions: vec![m.start()],
            });
        }

        candidates.into_values().collect()
    }
}

impl Default for M2MEntityExtractor {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_url() {
        let ext = M2MEntityExtractor::new();
        let results = ext.extract("Visit https://example.com for info");
        assert!(results.iter().any(|c| c.entity_type == "url"));
    }

    #[test]
    fn test_extract_date() {
        let ext = M2MEntityExtractor::new();
        let results = ext.extract("The event is on 2026-03-28");
        assert!(results.iter().any(|c| c.entity_type == "date" && c.text == "2026-03-28"));
    }

    #[test]
    fn test_extract_money() {
        let ext = M2MEntityExtractor::new();
        let results = ext.extract("Price is $1234");
        assert!(results.iter().any(|c| c.entity_type == "money"));
    }

    #[test]
    fn test_empty() {
        let ext = M2MEntityExtractor::new();
        let results = ext.extract("");
        assert!(results.is_empty());
    }
}

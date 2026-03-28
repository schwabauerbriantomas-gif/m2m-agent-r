//! Auto-Scaling for M2M — horizontal scaling management.
//! Ported from m2m-vector-search Python.

use serde::Serialize;

/// Node metrics for scaling decisions.
#[derive(Debug, Clone, Serialize)]
pub struct NodeMetrics {
    pub node_id: String,
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub qps: f64,
    pub latency_ms: f64,
    pub active_queries: usize,
}

/// Scaling decision.
#[derive(Debug, Clone, Serialize)]
pub struct ScalingDecision {
    pub action: String, // "scale_up" or "scale_down"
    pub reason: String,
    pub target_nodes: usize,
}

/// Scaling configuration.
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub cpu_threshold_up: f64,
    pub cpu_threshold_down: f64,
    pub latency_threshold_ms: f64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_nodes: 1,
            max_nodes: 10,
            cpu_threshold_up: 80.0,
            cpu_threshold_down: 20.0,
            latency_threshold_ms: 100.0,
        }
    }
}

/// Auto-scaler for M2M.
pub struct AutoScaler {
    config: ScalingConfig,
    current_nodes: usize,
    metrics_history: Vec<NodeMetrics>,
    scale_up_fn: Option<Box<dyn Fn() + Send + Sync>>,
    scale_down_fn: Option<Box<dyn Fn() + Send + Sync>>,
}

impl AutoScaler {
    pub fn new(min_nodes: usize, max_nodes: usize) -> Self {
        Self {
            config: ScalingConfig { min_nodes, max_nodes, ..Default::default() },
            current_nodes: min_nodes,
            metrics_history: Vec::new(),
            scale_up_fn: None,
            scale_down_fn: None,
        }
    }

    pub fn update_metrics(&mut self, metrics: NodeMetrics) {
        self.metrics_history.push(metrics);
        if self.metrics_history.len() > 100 {
            let start = self.metrics_history.len() - 100;
            self.metrics_history = self.metrics_history.split_off(start);
        }
    }

    pub fn evaluate_scaling(&self) -> Option<ScalingDecision> {
        if self.metrics_history.len() < 5 {
            return None;
        }

        let recent = &self.metrics_history[self.metrics_history.len() - 5..];
        let avg_cpu: f64 = recent.iter().map(|m| m.cpu_percent).sum::<f64>() / 5.0;
        let avg_latency: f64 = recent.iter().map(|m| m.latency_ms).sum::<f64>() / 5.0;

        if avg_cpu > self.config.cpu_threshold_up && self.current_nodes < self.config.max_nodes {
            return Some(ScalingDecision {
                action: "scale_up".into(),
                reason: format!("CPU avg {:.1}% > threshold {}", avg_cpu, self.config.cpu_threshold_up),
                target_nodes: self.current_nodes + 1,
            });
        }

        if avg_cpu < self.config.cpu_threshold_down && self.current_nodes > self.config.min_nodes {
            return Some(ScalingDecision {
                action: "scale_down".into(),
                reason: format!("CPU avg {:.1}% < threshold {}", avg_cpu, self.config.cpu_threshold_down),
                target_nodes: self.current_nodes - 1,
            });
        }

        None
    }

    pub fn execute_scaling(&mut self, decision: &ScalingDecision) {
        match decision.action.as_str() {
            "scale_up" => {
                self.current_nodes = decision.target_nodes;
                if let Some(ref f) = self.scale_up_fn { f(); }
            }
            "scale_down" => {
                self.current_nodes = decision.target_nodes;
                if let Some(ref f) = self.scale_down_fn { f(); }
            }
            _ => {}
        }
    }

    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "current_nodes": self.current_nodes,
            "min_nodes": self.config.min_nodes,
            "max_nodes": self.config.max_nodes,
            "metrics_samples": self.metrics_history.len(),
        })
    }

    pub fn current_nodes(&self) -> usize { self.current_nodes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_up_decision() {
        let mut scaler = AutoScaler::new(1, 5);
        for _ in 0..5 {
            scaler.update_metrics(NodeMetrics {
                node_id: "n1".into(),
                cpu_percent: 95.0,
                memory_percent: 80.0,
                qps: 1000.0,
                latency_ms: 50.0,
                active_queries: 100,
            });
        }
        let decision = scaler.evaluate_scaling().unwrap();
        assert_eq!(decision.action, "scale_up");
    }

    #[test]
    fn test_scale_down_decision() {
        let mut scaler = AutoScaler::new(1, 5);
        scaler.current_nodes = 2;
        for _ in 0..5 {
            scaler.update_metrics(NodeMetrics {
                node_id: "n1".into(),
                cpu_percent: 5.0,
                memory_percent: 30.0,
                qps: 10.0,
                latency_ms: 5.0,
                active_queries: 0,
            });
        }
        let decision = scaler.evaluate_scaling().unwrap();
        assert_eq!(decision.action, "scale_down");
    }

    #[test]
    fn test_no_decision_few_samples() {
        let scaler = AutoScaler::new(1, 5);
        assert!(scaler.evaluate_scaling().is_none());
    }
}

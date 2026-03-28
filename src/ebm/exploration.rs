//! EBM Exploration — exploration strategies for energy-based models.
//! Ported from m2m-vector-search Python.

use ndarray::Array1;

/// Exploration strategy.
#[derive(Debug, Clone, Copy)]
pub enum ExplorationStrategy {
    Random,
    EnergyGuided,
    DiversityMax,
    Curiosity,
}

/// EBM Explorer.
pub struct EbmExplorer {
    strategy: ExplorationStrategy,
    temperature: f64,
    history: Vec<(Array1<f32>, f64)>,
}

impl EbmExplorer {
    pub fn new(strategy: ExplorationStrategy, temperature: f64) -> Self {
        Self { strategy, temperature, history: Vec::new() }
    }

    /// Suggest exploration points based on strategy.
    pub fn suggest_exploration(&self, _current: &Array1<f32>, n: usize) -> Vec<Array1<f32>> {
        match self.strategy {
            ExplorationStrategy::Random => {
                // Random points in [-1, 1] range
                let dim = _current.len();
                (0..n).map(|_| {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    Array1::from_shape_vec(dim, (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).unwrap()
                }).collect()
            }
            _ => {
                // Fallback: random for other strategies
                let dim = _current.len();
                (0..n).map(|_| {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    Array1::from_shape_vec(dim, (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).unwrap()
                }).collect()
            }
        }
    }

    /// Record an exploration result.
    pub fn record(&mut self, point: Array1<f32>, energy: f64) {
        self.history.push((point, energy));
        if self.history.len() > 1000 {
            self.history = self.history.split_off(self.history.len() - 1000);
        }
    }

    pub fn history_len(&self) -> usize { self.history.len() }
}

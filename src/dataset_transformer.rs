//! Dataset Transformer — data transformations for M2M datasets.
//!
//! Handles normalization, splitting, augmentation, and format conversion.
//! Ported from m2m-vector-search Python.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Dataset split for train/val/test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub train: Array2<f32>,
    pub val: Option<Array2<f32>>,
    pub test: Option<Array2<f32>>,
    pub train_labels: Option<Vec<String>>,
    pub val_labels: Option<Vec<String>>,
    pub test_labels: Option<Vec<String>>,
}

/// Transformer configuration.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    pub normalize: bool,
    pub center: bool,
    pub unit_sphere: bool,
    pub augment_noise: f64,
    pub augment_flip: bool,
    pub train_ratio: f64,
    pub val_ratio: f64,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            center: true,
            unit_sphere: false,
            augment_noise: 0.0,
            augment_flip: true,
            train_ratio: 0.8,
            val_ratio: 0.1,
        }
    }
}

/// Dataset transformer for M2M.
pub struct DatasetTransformer {
    config: TransformConfig,
    mean: Option<Array1<f32>>,
    std: Option<Array1<f32>>,
}

impl DatasetTransformer {
    pub fn new(config: TransformConfig) -> Self {
        Self { config, mean: None, std: None }
    }

    /// Compute and store normalization statistics from data.
    pub fn fit(&mut self, data: &Array2<f32>) {
        let (n, d) = data.dim();
        if n == 0 { return; }

        if self.config.center {
            let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
            self.mean = Some(mean);
        }

        if self.config.normalize {
            let centered = match &self.mean {
                Some(m) => data - m,
                None => data.clone(),
            };
            let std = centered.std_axis(ndarray::Axis(0), 0.0);
            // Avoid division by zero
            let std = std.mapv(|v| if v < 1e-8 { 1.0 } else { v });
            self.std = Some(std);
        }
    }

    /// Apply stored transformations to data.
    pub fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        let mut result = data.clone();

        if let Some(ref mean) = self.mean {
            result = &result - mean;
        }

        if let Some(ref std) = self.std {
            result = result / std;
        }

        if self.config.unit_sphere {
            // Normalize each row to unit length
            for mut row in result.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
        }

        if self.config.augment_noise > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for mut row in result.rows_mut() {
                for val in row.iter_mut() {
                    let noise: f32 = rng.gen_range(-self.config.augment_noise..self.config.augment_noise) as f32;
                    *val += noise;
                }
            }
        }

        if self.config.augment_flip {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for mut row in result.rows_mut() {
                if rng.gen_bool(0.5) {
                    row.mapv_inplace(|x| -x);
                }
            }
        }

        result
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<f32>) -> Array2<f32> {
        self.fit(data);
        self.transform(data)
    }

    /// Split dataset into train/val/test.
    pub fn split(&self, data: &Array2<f32>, labels: Option<&[String]>) -> DatasetSplit {
        let n = data.nrows();
        let train_n = (n as f64 * self.config.train_ratio) as usize;
        let val_n = (n as f64 * self.config.val_ratio) as usize;
        let test_n = n - train_n - val_n;

        let train = data.slice(ndarray::s![..train_n, ..]).to_owned();
        let val = if val_n > 0 {
            Some(data.slice(ndarray::s![train_n..train_n + val_n, ..]).to_owned())
        } else {
            None
        };
        let test = if test_n > 0 {
            Some(data.slice(ndarray::s![train_n + val_n.., ..]).to_owned())
        } else {
            None
        };

        let (train_labels, val_labels, test_labels) = match labels {
            Some(l) => {
                let tl = l[..train_n].to_vec();
                let vl = if val_n > 0 { Some(l[train_n..train_n + val_n].to_vec()) } else { None };
                let tsl = if test_n > 0 { Some(l[train_n + val_n..].to_vec()) } else { None };
                (Some(tl), vl, tsl)
            }
            None => (None, None, None),
        };

        DatasetSplit { train, val, test, train_labels, val_labels, test_labels }
    }

    /// Get stored mean.
    pub fn get_mean(&self) -> Option<&Array1<f32>> {
        self.mean.as_ref()
    }

    /// Get stored std.
    pub fn get_std(&self) -> Option<&Array1<f32>> {
        self.std.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> Array2<f32> {
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()
    }

    #[test]
    fn test_fit_transform() {
        let config = TransformConfig::default();
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        assert_eq!(transformed.dim(), (4, 2));
        assert!(transformer.get_mean().is_some());
        assert!(transformer.get_std().is_some());
    }

    #[test]
    fn test_unit_sphere() {
        let config = TransformConfig { unit_sphere: true, center: false, normalize: false, ..Default::default() };
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        // Check each row is unit length
        for i in 0..transformed.nrows() {
            let row = transformed.row(i);
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Row {} norm = {}", i, norm);
        }
    }

    #[test]
    fn test_split() {
        let config = TransformConfig { train_ratio: 0.5, val_ratio: 0.25, ..Default::default() };
        let transformer = DatasetTransformer::new(config);
        let data = make_data();
        let split = transformer.split(&data, None);
        assert_eq!(split.train.nrows(), 2);
        assert!(split.val.is_some());
        assert!(split.test.is_some());
    }

    #[test]
    fn test_no_center() {
        let config = TransformConfig { center: false, normalize: false, augment_flip: false, ..Default::default() };
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        assert_eq!(&transformed, &data);
    }
}

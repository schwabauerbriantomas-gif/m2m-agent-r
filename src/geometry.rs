//! Geometry functions for M2M hypersphere operations.
//!
//! Optimized implementations using ndarray with iterators and zero-copy where possible.

use ndarray::{Array1, Array2, ArrayView1, Axis};

const EPSILON: f32 = 1e-8;
const DOT_CLIP_MARGIN: f32 = 1e-7;

/// Normalize vectors to unit hypersphere.
/// 
/// Divides each row by its L2 norm + epsilon to avoid division by zero.
/// 
/// # Arguments
/// * `x` - Input array of shape (N, D) or (D,)
/// 
/// # Returns
/// Normalized array with same shape
pub fn normalize_sphere(x: &Array2<f32>) -> Array2<f32> {
    let n_rows = x.nrows();
    let mut result = x.clone();
    
    for i in 0..n_rows {
        let row = result.row(i);
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt() + EPSILON;
        let mut row_mut = result.row_mut(i);
        row_mut.mapv_inplace(|v| v / norm);
    }
    
    result
}

/// Normalize a single vector to unit hypersphere.
pub fn normalize_sphere_1d(x: &ArrayView1<f32>) -> Array1<f32> {
    let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt() + EPSILON;
    x.mapv(|v| v / norm)
}

/// Calculate geodesic distance between vectors.
/// 
/// Returns arccos of the clipped dot product, ensuring numerical stability.
/// 
/// # Arguments
/// * `x` - First vector
/// * `y` - Second vector
/// 
/// # Returns
/// Geodesic distance in radians
pub fn geodesic_distance(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    let dot: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    
    // Early exit for near-identical vectors
    if dot > 1.0 - 1e-5 {
        return 0.0;
    }
    
    // Clip to avoid numerical issues with arccos
    let clipped = dot.clamp(-1.0 + DOT_CLIP_MARGIN, 1.0 - DOT_CLIP_MARGIN);
    
    clipped.acos()
}

/// Calculate geodesic distances between pairs of vectors.
/// 
/// # Arguments
/// * `x` - First array of shape (N, D)
/// * `y` - Second array of shape (N, D)
/// 
/// # Returns
/// Array of shape (N,) with pairwise distances
pub fn geodesic_distance_batch(x: &Array2<f32>, y: &Array2<f32>) -> Array1<f32> {
    assert_eq!(x.nrows(), y.nrows(), "Input arrays must have same number of rows");
    assert_eq!(x.ncols(), y.ncols(), "Input arrays must have same number of columns");
    
    x.axis_iter(Axis(0))
        .zip(y.axis_iter(Axis(0)))
        .map(|(row_x, row_y)| geodesic_distance(&row_x, &row_y))
        .collect()
}

/// Exponential map on sphere.
/// 
/// Currently returns identity (simplified implementation matching Python).
/// TODO: Implement proper exponential map for hypersphere.
#[inline(always)]
pub fn exp_map<T: std::clone::Clone>(x: T) -> T {
    x
}

/// Logarithmic map on sphere.
/// 
/// Currently returns identity (simplified implementation matching Python).
/// TODO: Implement proper logarithmic map for hypersphere.
#[inline(always)]
pub fn log_map<T: std::clone::Clone>(x: T) -> T {
    x
}

/// Project vector to tangent space.
/// 
/// Currently returns identity (simplified implementation matching Python).
/// TODO: Implement proper tangent space projection.
#[inline(always)]
pub fn project_to_tangent<T: std::clone::Clone>(v: T) -> T {
    v
}

/// Compute pairwise cosine similarity matrix.
/// 
/// # Arguments
/// * `x` - Input array of shape (N, D), assumed to be normalized
/// 
/// # Returns
/// Similarity matrix of shape (N, N)
pub fn cosine_similarity_matrix(x: &Array2<f32>) -> Array2<f32> {
    let n = x.nrows();
    let mut result = Array2::<f32>::zeros((n, n));
    
    // Compute upper triangle and mirror
    for i in 0..n {
        for j in i..n {
            let dot: f32 = x.row(i).iter()
                .zip(x.row(j).iter())
                .map(|(&a, &b)| a * b)
                .sum();
            
            result[[i, j]] = dot;
            if i != j {
                result[[j, i]] = dot;
            }
        }
    }
    
    result
}

/// Find k nearest neighbors by geodesic distance.
/// 
/// # Arguments
/// * `query` - Query vector
/// * `database` - Database of vectors (N, D)
/// * `k` - Number of neighbors
/// 
/// # Returns
/// Vector of (index, distance) pairs sorted by distance
pub fn knn_geodesic(query: &ArrayView1<f32>, database: &Array2<f32>, k: usize) -> Vec<(usize, f32)> {
    let mut distances: Vec<(usize, f32)> = database
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(idx, row)| (idx, geodesic_distance(query, &row)))
        .collect();
    
    // Partial sort to get top k
    let k = k.min(distances.len());
    distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize_sphere() {
        let x = array![[3.0_f32, 4.0_f32], [0.0_f32, 0.0_f32]];
        let normalized = normalize_sphere(&x);
        
        // First row: [3, 4] -> [3/5, 4/5]
        assert_relative_eq!(normalized[[0, 0]], 0.6_f32, epsilon = 1e-5);
        assert_relative_eq!(normalized[[0, 1]], 0.8_f32, epsilon = 1e-5);
        
        // Second row: [0, 0] -> [0, 0] (divided by epsilon)
        assert_relative_eq!(normalized[[1, 0]], 0.0_f32, epsilon = 1e-5);
        assert_relative_eq!(normalized[[1, 1]], 0.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_geodesic_distance() {
        let x = array![1.0_f32, 0.0_f32, 0.0_f32];
        let y = array![0.0_f32, 1.0_f32, 0.0_f32];
        
        let dist = geodesic_distance(&x.view(), &y.view());
        assert_relative_eq!(dist, std::f32::consts::FRAC_PI_2, epsilon = 1e-5);
        
        // Same vector should have distance 0
        let dist_same = geodesic_distance(&x.view(), &x.view());
        assert_relative_eq!(dist_same, 0.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_identity_maps() {
        let x = array![1.0_f32, 2.0_f32, 3.0_f32];
        assert_eq!(exp_map(x.clone()), x);
        assert_eq!(log_map(x.clone()), x);
        assert_eq!(project_to_tangent(x.clone()), x);
    }
}

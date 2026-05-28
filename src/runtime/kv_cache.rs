// src/runtime/kv_cache.rs
use crate::matrix::Matrix;

#[derive(Debug, PartialEq, Eq)]
pub enum KVCacheError {
    CacheFull,
}

pub struct QKVCache<const MAX_SEQ: usize, const DIM: usize> {
    pub keys: Matrix<i8, MAX_SEQ, DIM>,
    pub values: Matrix<i8, MAX_SEQ, DIM>,
    pub key_scale: f32,
    pub value_scale: f32,
    pub current_len: usize,
}

impl<const MAX_SEQ: usize, const DIM: usize> QKVCache<MAX_SEQ, DIM> {
    pub fn new() -> Self {
        Self {
            keys: Matrix::zeros(),
            values: Matrix::zeros(),
            key_scale: 1.0,
            value_scale: 1.0,
            current_len: 0,
        }
    }

    pub fn append(&mut self, key: &[i8; DIM], value: &[i8; DIM], k_scale: f32, v_scale: f32) -> Result<(), KVCacheError> {
        if self.is_full() {
            return Err(KVCacheError::CacheFull);
        }
        
        let idx = self.current_len;
        for i in 0..DIM {
            self.keys.data[idx][i] = key[i];
            self.values.data[idx][i] = value[i];
        }
        self.key_scale = k_scale;
        self.value_scale = v_scale;
        self.current_len += 1;
        
        Ok(())
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn is_full(&self) -> bool {
        self.current_len >= MAX_SEQ
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn get_active_keys(&self) -> Matrix<i8, MAX_SEQ, DIM> {
        self.keys.clone()
    }

    pub fn get_active_values(&self) -> Matrix<i8, MAX_SEQ, DIM> {
        self.values.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append() {
        let mut cache = QKVCache::<4, 2>::new();
        assert_eq!(cache.current_len(), 0);
        
        cache.append(&[1, 2], &[3, 4], 1.0, 1.0).unwrap();
        assert_eq!(cache.current_len(), 1);
        
        let keys = cache.get_active_keys();
        assert_eq!(keys.data[0], [1, 2]);
    }

    #[test]
    fn test_kv_cache_bounds() {
        let mut cache = QKVCache::<2, 2>::new();
        cache.append(&[1, 2], &[3, 4], 1.0, 1.0).unwrap();
        cache.append(&[5, 6], &[7, 8], 1.0, 1.0).unwrap();
        
        let res = cache.append(&[9, 10], &[11, 12], 1.0, 1.0);
        assert_eq!(res, Err(KVCacheError::CacheFull));
    }

    #[test]
    fn test_kv_cache_reuse() {
        let mut cache = QKVCache::<4, 2>::new();
        cache.append(&[1, 2], &[3, 4], 1.0, 1.0).unwrap();
        cache.append(&[5, 6], &[7, 8], 1.0, 1.0).unwrap();
        
        let keys = cache.get_active_keys();
        assert_eq!(keys.data[0], [1, 2]);
        assert_eq!(keys.data[1], [5, 6]);
    }
}

// src/model/tensor.rs
use crate::matrix::Matrix;
use crate::model::format::ModelLoadError;

pub struct BufferReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BufferReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub fn advance(&mut self, len: usize) -> Result<&'a [u8], ModelLoadError> {
        if self.offset + len > self.bytes.len() {
            return Err(ModelLoadError::BufferTooSmall);
        }
        let slice = &self.bytes[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    pub fn extract_matrix<const R: usize, const C: usize>(&mut self) -> Result<Matrix<i8, R, C>, ModelLoadError> {
        let len = R * C;
        let slice = self.advance(len)?;
        let mut mat = Matrix::<i8, R, C>::zeros();
        
        let mut i = 0;
        for r in 0..R {
            for c in 0..C {
                mat.data[r][c] = slice[i] as i8;
                i += 1;
            }
        }
        Ok(mat)
    }

    pub fn extract_array<const N: usize>(&mut self) -> Result<[i8; N], ModelLoadError> {
        let slice = self.advance(N)?;
        let mut arr = [0; N];
        for i in 0..N {
            arr[i] = slice[i] as i8;
        }
        Ok(arr)
    }

    pub fn extract_f32(&mut self) -> Result<f32, ModelLoadError> {
        let slice = self.advance(4)?;
        let bytes: [u8; 4] = slice.try_into().map_err(|_| ModelLoadError::BufferTooSmall)?;
        Ok(f32::from_le_bytes(bytes))
    }

    pub fn extract_f32_array<const N: usize>(&mut self) -> Result<[f32; N], ModelLoadError> {
        let mut arr = [0.0; N];
        for i in 0..N {
            arr[i] = self.extract_f32()?;
        }
        Ok(arr)
    }
}

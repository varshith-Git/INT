// src/model/format.rs

#[derive(Debug, PartialEq, Eq)]
pub enum ModelLoadError {
    InvalidMagic,
    UnsupportedVersion,
    BufferTooSmall,
    ShapeMismatch,
    InvalidTensorLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QModelHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub vocab: u32,
    pub seq: u32,
    pub dim: u32,
    pub hidden: u32,
}

impl QModelHeader {
    pub const SIZE: usize = 24;

    pub fn parse(bytes: &[u8]) -> Result<Self, ModelLoadError> {
        if bytes.len() < Self::SIZE {
            return Err(ModelLoadError::BufferTooSmall);
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != b"QMOD" {
            return Err(ModelLoadError::InvalidMagic);
        }

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return Err(ModelLoadError::UnsupportedVersion);
        }

        let vocab = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let seq = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let dim = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let hidden = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);

        Ok(Self { magic, version, vocab, seq, dim, hidden })
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.vocab.to_le_bytes());
        buf[12..16].copy_from_slice(&self.seq.to_le_bytes());
        buf[16..20].copy_from_slice(&self.dim.to_le_bytes());
        buf[20..24].copy_from_slice(&self.hidden.to_le_bytes());
        buf
    }

    pub fn expected_model_size(&self) -> usize {
        let v = self.vocab as usize;
        let d = self.dim as usize;
        let h = self.hidden as usize;

        Self::SIZE +
        (v * d) + // embedding
        (d * d) + // q_proj
        (d * d) + // k_proj
        (d * d) + // v_proj
        d +       // norm1_weight
        (d * h) + // ff1
        (h * d) + // ff2
        d +       // norm2_weight
        (d * v)   // output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QTransformerHeader {
    pub magic: [u8; 4], // b"QTRN"
    pub version: u32,
    pub vocab_size: u32,
    pub seq_len: u32,
    pub dim: u32,
    pub heads: u32,
    pub layers: u32,
    pub hidden: u32,
}

impl QTransformerHeader {
    pub const SIZE: usize = 32;

    pub fn parse(bytes: &[u8]) -> Result<Self, ModelLoadError> {
        if bytes.len() < Self::SIZE {
            return Err(ModelLoadError::BufferTooSmall);
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != b"QTRN" {
            return Err(ModelLoadError::InvalidMagic);
        }

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return Err(ModelLoadError::UnsupportedVersion);
        }

        let vocab_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let seq_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let dim = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let heads = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let layers = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        let hidden = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);

        Ok(Self { magic, version, vocab_size, seq_len, dim, heads, layers, hidden })
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.vocab_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[16..20].copy_from_slice(&self.dim.to_le_bytes());
        buf[20..24].copy_from_slice(&self.heads.to_le_bytes());
        buf[24..28].copy_from_slice(&self.layers.to_le_bytes());
        buf[28..32].copy_from_slice(&self.hidden.to_le_bytes());
        buf
    }
}

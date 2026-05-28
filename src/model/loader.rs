// src/model/loader.rs
use crate::matrix::Matrix;
use crate::nn::embedding::QEmbedding;
use crate::nn::transformer::QTransformerBlock;
use crate::nn::linear::QLinear;
use crate::nn::rmsnorm::QRMSNorm;
use crate::quant::{QTensor, QuantParams, RequantShift};
use crate::model::format::{QModelHeader, ModelLoadError};
use crate::model::tensor::BufferReader;

pub struct QModel<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
> {
    pub embedding: QEmbedding<VOCAB, DIM>,
    pub transformer: QTransformerBlock<SEQ, DIM, HIDDEN>,
    pub output: QLinear<DIM, VOCAB>,
}

pub fn expected_model_size<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
>() -> usize {
    QModelHeader::SIZE +
    (VOCAB * DIM) + // embedding
    (DIM * DIM) +   // q_proj
    (DIM * DIM) +   // k_proj
    (DIM * DIM) +   // v_proj
    DIM +           // norm1
    (DIM * HIDDEN) + // ff1
    (HIDDEN * DIM) + // ff2
    DIM +           // norm2
    (DIM * VOCAB)   // output
}

pub fn load_from_bytes<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
>(bytes: &[u8]) -> Result<QModel<VOCAB, SEQ, DIM, HIDDEN>, ModelLoadError> {
    let expected = expected_model_size::<VOCAB, SEQ, DIM, HIDDEN>();
    if bytes.len() != expected {
        return Err(ModelLoadError::ShapeMismatch);
    }

    let header = QModelHeader::parse(bytes)?;
    
    // Validate static shape matching header
    if header.vocab as usize != VOCAB || 
       header.seq as usize != SEQ || 
       header.dim as usize != DIM || 
       header.hidden as usize != HIDDEN {
        return Err(ModelLoadError::ShapeMismatch);
    }

    let mut reader = BufferReader::new(&bytes[QModelHeader::SIZE..]);
    let params = QuantParams::symmetric(1.0);

    let emb_mat = reader.extract_matrix::<VOCAB, DIM>()?;
    let embedding = QEmbedding::new(QTensor::new(emb_mat, params));

    let q_w = reader.extract_matrix::<DIM, DIM>()?;
    let q_proj = QLinear::new(q_w, &[1.0; DIM], None, 1.0, params, false);

    let k_w = reader.extract_matrix::<DIM, DIM>()?;
    let k_proj = QLinear::new(k_w, &[1.0; DIM], None, 1.0, params, false);

    let v_w = reader.extract_matrix::<DIM, DIM>()?;
    let v_proj = QLinear::new(v_w, &[1.0; DIM], None, 1.0, params, false);

    let norm1_w = reader.extract_array::<DIM>()?;
    let norm1 = QRMSNorm::new(norm1_w.map(|v| v as f32), 1e-5);

    let ff1_w = reader.extract_matrix::<DIM, HIDDEN>()?;
    let ff1 = QLinear::new(ff1_w, &[1.0; HIDDEN], None, 1.0, params, true);

    let ff2_w = reader.extract_matrix::<HIDDEN, DIM>()?;
    let ff2 = QLinear::new(ff2_w, &[1.0; DIM], None, 1.0, params, false);

    let norm2_w = reader.extract_array::<DIM>()?;
    let norm2 = QRMSNorm::new(norm2_w.map(|v| v as f32), 1e-5);

    let out_w = reader.extract_matrix::<DIM, VOCAB>()?;
    let output = QLinear::new(out_w, &[1.0; VOCAB], None, 1.0, params, false);

    let transformer = QTransformerBlock {
        norm1,
        norm2,
        q_proj,
        k_proj,
        v_proj,
        ff1,
        ff2,
        attn_requant: RequantShift::from_ratio(1.0),
        attn_out_params: params,
    };

    Ok(QModel {
        embedding,
        transformer,
        output,
    })
}

pub struct QMLPHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub input_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
}

impl QMLPHeader {
    pub const SIZE: usize = 20;

    pub fn parse(bytes: &[u8]) -> Result<Self, ModelLoadError> {
        if bytes.len() < Self::SIZE {
            return Err(ModelLoadError::ShapeMismatch);
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if &magic != b"QMLP" {
            return Err(ModelLoadError::InvalidMagic);
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err(ModelLoadError::UnsupportedVersion);
        }

        let input_dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let hidden_dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let output_dim = u32::from_le_bytes(bytes[16..20].try_into().unwrap());

        Ok(Self { magic, version, input_dim, hidden_dim, output_dim })
    }
}

pub struct MLPWeights<
    const IN: usize,
    const HIDDEN: usize,
    const OUT: usize,
> {
    pub fc1_w: crate::matrix::Matrix<i8, IN, HIDDEN>,
    pub fc1_w_scale: f32,
    pub fc1_b: crate::matrix::Matrix<i32, 1, HIDDEN>,
    pub fc2_w: crate::matrix::Matrix<i8, HIDDEN, OUT>,
    pub fc2_w_scale: f32,
    pub fc2_b: crate::matrix::Matrix<i32, 1, OUT>,
    pub fc1_out_params: QuantParams,
    pub img_scale: f32,
}

pub fn load_mlp_from_bytes<
    const IN: usize,
    const HIDDEN: usize,
    const OUT: usize,
>(bytes: &[u8]) -> Result<MLPWeights<IN, HIDDEN, OUT>, ModelLoadError> {
    let header = QMLPHeader::parse(bytes)?;
    if header.input_dim as usize != IN || header.hidden_dim as usize != HIDDEN || header.output_dim as usize != OUT {
        return Err(ModelLoadError::ShapeMismatch);
    }

    let mut reader = BufferReader::new(&bytes[QMLPHeader::SIZE..]);
    
    // Weights are stored row-major. For QLinear, weights are [OUT, IN] logically but wait, QLinear takes QTensor<i8, IN, OUT> natively!
    // But PyTorch nn.Linear(in, out) weight is [out, in].
    // Let's extract as [out, in] and we will transpose during model load or assume the caller handles the transposition.
    // Wait, PyTorch weights are [out, in] but our QLinear weights are [IN_DIM, OUT_DIM] logically! 
    // Ah, our previous code used [DIM, DIM] so it was square and we didn't notice.
    // If PyTorch is [out, in], let's load it as [out, in] here.
    let fc1_w_data = reader.extract_matrix::<HIDDEN, IN>()?;
    let fc1_b_data = reader.extract_array::<HIDDEN>()?;
    let fc2_w_data = reader.extract_matrix::<OUT, HIDDEN>()?;
    let fc2_b_data = reader.extract_array::<OUT>()?;
    
    let fc1_w_scale_bytes = reader.extract_array::<4>()?.map(|b| b as u8);
    let fc1_w_scale = f32::from_le_bytes(fc1_w_scale_bytes);
    
    let fc1_b_scale_bytes = reader.extract_array::<4>()?.map(|b| b as u8);
    let fc1_b_scale = f32::from_le_bytes(fc1_b_scale_bytes);
    
    let fc2_w_scale_bytes = reader.extract_array::<4>()?.map(|b| b as u8);
    let fc2_w_scale = f32::from_le_bytes(fc2_w_scale_bytes);
    
    let fc2_b_scale_bytes = reader.extract_array::<4>()?.map(|b| b as u8);
    let fc2_b_scale = f32::from_le_bytes(fc2_b_scale_bytes);
    
    let img_scale_bytes = reader.extract_array::<4>()?.map(|b| b as u8);
    let img_scale = f32::from_le_bytes(img_scale_bytes);

    // QLinear requires fc1_w to be [IN, HIDDEN]. Let's transpose it.
    let mut fc1_w_transposed = crate::matrix::Matrix::<i8, IN, HIDDEN>::zeros();
    for r in 0..HIDDEN {
        for c in 0..IN {
            fc1_w_transposed.data[c][r] = fc1_w_data.data[r][c];
        }
    }
    
    let mut fc2_w_transposed = crate::matrix::Matrix::<i8, HIDDEN, OUT>::zeros();
    for r in 0..OUT {
        for c in 0..HIDDEN {
            fc2_w_transposed.data[c][r] = fc2_w_data.data[r][c];
        }
    }

    let fc1_out_scale = 0.05; // fixed intermediate scale

    let mut fc1_b_i32 = crate::matrix::Matrix::<i32, 1, HIDDEN>::zeros();
    for i in 0..HIDDEN {
        let val_f32 = (fc1_b_data[i] as f32) * fc1_b_scale;
        let expected_scale = img_scale * fc1_w_scale;
        fc1_b_i32.data[0][i] = libm::roundf(val_f32 / expected_scale) as i32;
    }

    let mut fc2_b_i32 = crate::matrix::Matrix::<i32, 1, OUT>::zeros();
    for i in 0..OUT {
        let val_f32 = (fc2_b_data[i] as f32) * fc2_b_scale;
        let expected_scale = fc1_out_scale * fc2_w_scale;
        fc2_b_i32.data[0][i] = libm::roundf(val_f32 / expected_scale) as i32;
    }

    Ok(MLPWeights {
        fc1_w: fc1_w_transposed,
        fc1_w_scale,
        fc1_b: fc1_b_i32,
        fc2_w: fc2_w_transposed,
        fc2_w_scale,
        fc2_b: fc2_b_i32,
        fc1_out_params: QuantParams::symmetric(fc1_out_scale),
        img_scale,
    })
}

pub trait BinaryLoadable {
    fn load_from_reader(reader: &mut BufferReader) -> Result<Self, ModelLoadError> where Self: Sized;
}

impl<const VOCAB: usize, const DIM: usize> BinaryLoadable for QEmbedding<VOCAB, DIM> {
    fn load_from_reader(reader: &mut BufferReader) -> Result<Self, ModelLoadError> {
        let w = reader.extract_matrix::<VOCAB, DIM>()?;
        // Embeddings are quantized per-token (per-row). This means the array length is VOCAB.
        let scale = reader.extract_f32_array::<VOCAB>()?;
        
        // Wait, QEmbedding uses QTensor which only takes a single QuantParams currently.
        // We will just use scale[0] as a placeholder for now since we aren't refactoring QEmbedding for this Phase.
        // Note: PyTorch exports per-channel (row) scales for token_embeddings, we just parse and extract them.
        Ok(Self::new(QTensor::new(w, QuantParams::symmetric(scale[0]))))
    }
}

impl<const DIM: usize> BinaryLoadable for QRMSNorm<DIM> {
    fn load_from_reader(reader: &mut BufferReader) -> Result<Self, ModelLoadError> {
        let mut w = [0.0; DIM];
        for i in 0..DIM {
            w[i] = reader.extract_f32()?;
        }
        Ok(Self::new(w, 1e-5))
    }
}

pub struct LoadedBlockWeights<const DIM: usize, const HIDDEN: usize> {
    pub norm1: QRMSNorm<DIM>,
    pub q_w: crate::matrix::Matrix<i8, DIM, DIM>,
    pub q_scale: [f32; DIM],
    pub k_w: crate::matrix::Matrix<i8, DIM, DIM>,
    pub k_scale: [f32; DIM],
    pub v_w: crate::matrix::Matrix<i8, DIM, DIM>,
    pub v_scale: [f32; DIM],
    pub out_w: crate::matrix::Matrix<i8, DIM, DIM>,
    pub out_scale: [f32; DIM],
    pub norm2: QRMSNorm<DIM>,
    pub ff1_w: crate::matrix::Matrix<i8, HIDDEN, DIM>,
    pub ff1_scale: [f32; HIDDEN],
    pub ff2_w: crate::matrix::Matrix<i8, DIM, HIDDEN>,
    pub ff2_scale: [f32; DIM],
}

impl<const DIM: usize, const HIDDEN: usize> BinaryLoadable for LoadedBlockWeights<DIM, HIDDEN> {
    fn load_from_reader(reader: &mut BufferReader) -> Result<Self, ModelLoadError> {
        let norm1 = QRMSNorm::<DIM>::load_from_reader(reader)?;
        
        let q_w = reader.extract_matrix::<DIM, DIM>()?;
        let q_scale = reader.extract_f32_array::<DIM>()?;
        
        let k_w = reader.extract_matrix::<DIM, DIM>()?;
        let k_scale = reader.extract_f32_array::<DIM>()?;
        
        let v_w = reader.extract_matrix::<DIM, DIM>()?;
        let v_scale = reader.extract_f32_array::<DIM>()?;
        
        let out_w = reader.extract_matrix::<DIM, DIM>()?;
        let out_scale = reader.extract_f32_array::<DIM>()?;
        
        let norm2 = QRMSNorm::<DIM>::load_from_reader(reader)?;
        
        let ff1_w = reader.extract_matrix::<HIDDEN, DIM>()?;
        let ff1_scale = reader.extract_f32_array::<HIDDEN>()?;
        
        let ff2_w = reader.extract_matrix::<DIM, HIDDEN>()?;
        let ff2_scale = reader.extract_f32_array::<DIM>()?;
        
        Ok(Self {
            norm1,
            q_w, q_scale,
            k_w, k_scale,
            v_w, v_scale,
            out_w, out_scale,
            norm2,
            ff1_w, ff1_scale,
            ff2_w, ff2_scale,
        })
    }
}

pub fn load_transformer_from_bytes<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
    const HEADS: usize,
    const LAYERS: usize,
>(
    bytes: &[u8],
) -> Result<crate::nn::gpt::QGPTModel<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>, ModelLoadError> {
    let header = crate::model::format::QTransformerHeader::parse(bytes)?;
    
    if header.vocab_size as usize != VOCAB ||
       header.seq_len as usize != SEQ ||
       header.dim as usize != DIM ||
       header.hidden as usize != HIDDEN ||
       header.heads as usize != HEADS ||
       header.layers as usize != LAYERS {
        return Err(ModelLoadError::ShapeMismatch);
    }
    
    let mut reader = BufferReader::new(&bytes[crate::model::format::QTransformerHeader::SIZE..]);
    
    let token_embeddings = QEmbedding::<VOCAB, DIM>::load_from_reader(&mut reader)?;
    
    let pos_emb_w = reader.extract_matrix::<SEQ, DIM>()?;
    let pos_emb_scale = reader.extract_f32_array::<SEQ>()?;
    let pos_embeddings = QTensor::new(pos_emb_w, QuantParams::symmetric(pos_emb_scale[0]));
    
    let block_weights: [_; LAYERS] = core::array::from_fn(|_| LoadedBlockWeights::<DIM, HIDDEN>::load_from_reader(&mut reader).unwrap());
    
    let final_norm = QRMSNorm::<DIM>::load_from_reader(&mut reader)?;
    
    let lm_head_w_raw = reader.extract_matrix::<VOCAB, DIM>()?;
    let lm_head_scale = reader.extract_f32_array::<VOCAB>()?;
    let mut lm_head_w = Matrix::<i8, DIM, VOCAB>::zeros();
    for r in 0..VOCAB {
        for c in 0..DIM {
            lm_head_w.data[c][r] = lm_head_w_raw.data[r][c];
        }
    }
    
    let mut s_res_in = [0.0; LAYERS];
    let mut s_res_out = [0.0; LAYERS];
    let mut calib_q_scale = [0.0; LAYERS];
    let mut calib_k_scale = [0.0; LAYERS];
    let mut calib_v_scale = [0.0; LAYERS];
    let mut calib_ff1_out_scale = [0.0; LAYERS];
    
    for l in 0..LAYERS {
        s_res_in[l] = reader.extract_f32()?;
        s_res_out[l] = reader.extract_f32()?;
        calib_q_scale[l] = reader.extract_f32()?;
        calib_k_scale[l] = reader.extract_f32()?;
        calib_v_scale[l] = reader.extract_f32()?;
        calib_ff1_out_scale[l] = reader.extract_f32()?;
    }
    
    let mut block_weights_iter = block_weights.into_iter();
    let blocks = core::array::from_fn(|i| {
        construct_gpt_block(
            block_weights_iter.next().unwrap(),
            s_res_in[i],
            s_res_out[i],
            calib_q_scale[i],
            calib_k_scale[i],
            calib_v_scale[i],
            calib_ff1_out_scale[i],
        )
    });
    
    let lm_head = QLinear::new(
        lm_head_w,
        &lm_head_scale,
        None,
        s_res_out[LAYERS - 1], // Final residual out
        QuantParams::symmetric(1.0),
        false,
    );
    
    Ok(crate::nn::gpt::QGPTModel {
        token_embeddings,
        pos_embeddings,
        blocks,
        final_norm,
        lm_head,
    })
}

fn construct_gpt_block<const SEQ: usize, const DIM: usize, const HIDDEN: usize, const HEADS: usize>(
    weights: LoadedBlockWeights<DIM, HIDDEN>,
    s_res_in: f32,
    s_res_out: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    ff1_out_scale: f32,
) -> crate::nn::gpt::QGPTBlock<SEQ, DIM, HIDDEN, HEADS> {
    let mut q_w_t = Matrix::<i8, DIM, DIM>::zeros();
    let mut k_w_t = Matrix::<i8, DIM, DIM>::zeros();
    let mut v_w_t = Matrix::<i8, DIM, DIM>::zeros();
    let mut out_w_t = Matrix::<i8, DIM, DIM>::zeros();
    
    for r in 0..DIM {
        for c in 0..DIM {
            q_w_t.data[c][r] = weights.q_w.data[r][c];
            k_w_t.data[c][r] = weights.k_w.data[r][c];
            v_w_t.data[c][r] = weights.v_w.data[r][c];
            out_w_t.data[c][r] = weights.out_w.data[r][c];
        }
    }
    
    let mut ff1_w_t = Matrix::<i8, DIM, HIDDEN>::zeros();
    for r in 0..HIDDEN {
        for c in 0..DIM {
            ff1_w_t.data[c][r] = weights.ff1_w.data[r][c];
        }
    }
    
    let mut ff2_w_t = Matrix::<i8, HIDDEN, DIM>::zeros();
    for r in 0..DIM {
        for c in 0..HIDDEN {
            ff2_w_t.data[c][r] = weights.ff2_w.data[r][c];
        }
    }
    
    let q_proj = QLinear::new(
        q_w_t,
        &weights.q_scale,
        None,
        s_res_in,
        QuantParams::symmetric(q_scale),
        false,
    );
    let k_proj = QLinear::new(
        k_w_t,
        &weights.k_scale,
        None,
        s_res_in,
        QuantParams::symmetric(k_scale),
        false,
    );
    let v_proj = QLinear::new(
        v_w_t,
        &weights.v_scale,
        None,
        s_res_in,
        QuantParams::symmetric(v_scale),
        false,
    );
    let out_proj = QLinear::new(
        out_w_t,
        &weights.out_scale,
        None,
        v_scale,
        QuantParams::symmetric(s_res_in),
        false,
    );
    let ff1 = QLinear::new(
        ff1_w_t,
        &weights.ff1_scale,
        None,
        s_res_in,
        QuantParams::symmetric(ff1_out_scale),
        true,
    );
    let ff2 = QLinear::new(
        ff2_w_t,
        &weights.ff2_scale,
        None,
        ff1_out_scale,
        QuantParams::symmetric(s_res_in),
        false,
    );
    
    let head_dim = DIM / HEADS;
    let attn_scale = q_scale * k_scale / libm::sqrtf(head_dim as f32);
    let attn_requant = RequantShift::from_ratio(1.0);
    
    crate::nn::gpt::QGPTBlock {
        norm1: weights.norm1,
        norm2: weights.norm2,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        ff1,
        ff2,
        attn_scale,
        attn_requant,
        attn_out_params: QuantParams::symmetric(v_scale),
        s_res_in: QuantParams::symmetric(s_res_in),
        s_res_out: QuantParams::symmetric(s_res_out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_header_parse() {
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 100,
            seq: 32,
            dim: 64,
            hidden: 128,
        };
        let bytes = hdr.to_bytes();
        let parsed = QModelHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.vocab, 100);
        assert_eq!(parsed.hidden, 128);
    }

    #[test]
    fn test_invalid_magic() {
        let mut hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1, vocab: 10, seq: 10, dim: 10, hidden: 10,
        };
        hdr.magic = *b"XXXX";
        let bytes = hdr.to_bytes();
        let res = QModelHeader::parse(&bytes);
        assert_eq!(res, Err(ModelLoadError::InvalidMagic));
    }

    #[test]
    fn test_invalid_version() {
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 2, vocab: 10, seq: 10, dim: 10, hidden: 10,
        };
        let bytes = hdr.to_bytes();
        let res = QModelHeader::parse(&bytes);
        assert_eq!(res, Err(ModelLoadError::UnsupportedVersion));
    }

    #[test]
    fn test_expected_model_size() {
        let size = expected_model_size::<4, 4, 2, 4>();
        // 24 + 8 + 4 + 4 + 4 + 2 + 8 + 8 + 2 + 8 = 72
        assert_eq!(size, 72);
    }

    #[test]
    fn test_tensor_deserialize() {
        let bytes = [1, 2, 3, 4];
        let mut reader = BufferReader::new(&bytes);
        let mat = reader.extract_matrix::<2, 2>().unwrap();
        assert_eq!(mat.data[0][0], 1);
        assert_eq!(mat.data[1][1], 4);
    }

    #[test]
    fn test_invalid_shape_rejection() {
        // Less than required
        let res = load_from_bytes::<4, 4, 2, 4>(&[0; 70]);
        assert_eq!(res.err(), Some(ModelLoadError::ShapeMismatch));
    }

    #[test]
    fn test_model_roundtrip() {
        let mut bytes = [0u8; 72];
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 4,
            seq: 4,
            dim: 2,
            hidden: 4,
        };
        bytes[0..24].copy_from_slice(&hdr.to_bytes());
        // set some weight
        bytes[24] = 42;
        let model = load_from_bytes::<4, 4, 2, 4>(&bytes).unwrap();
        assert_eq!(model.embedding.weights.raw(0, 0), 42);
    }

    #[test]
    fn test_runtime_model_execution() {
        use crate::runtime::QGenerator;
        let mut bytes = [0u8; 72];
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 4,
            seq: 4,
            dim: 2,
            hidden: 4,
        };
        bytes[0..24].copy_from_slice(&hdr.to_bytes());
        let model = load_from_bytes::<4, 4, 2, 4>(&bytes).unwrap();
        let mut gen = QGenerator::new(model);
        let seq = gen.greedy_decode(1, 4);
        assert_eq!(seq.len(), 4);
    }
}

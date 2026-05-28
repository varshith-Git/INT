import sys

content = open("src/nn/gpt.rs").read()

old = '''use crate::nn::transformer::requantize_attention_output_f32;'''
new = '''use crate::nn::transformer::requantize_attention_output_f32;
#[cfg(feature = "std")]
use std::eprintln;'''
assert old in content, "Failed old"
content = content.replace(old, new, 1)

open("src/nn/gpt.rs", "w").write(content)
print("gpt.rs patched with use std::eprintln")

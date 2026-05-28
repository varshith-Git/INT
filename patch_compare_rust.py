import sys

content = open("examples/compare_rust.rs").read()

old = '''    let vocab_str = fs::read_to_string("artifacts/vocab.json").unwrap();
    let vocab: std::collections::HashMap<char, usize> = serde_json::from_str(&vocab_str).unwrap();

    let prompt = "ACT I\\n";
    let mut tokens = [0usize; SEQ];
    for (i, c) in prompt.chars().enumerate() {
        tokens[i] = *vocab.get(&c).unwrap();
    }'''
new = '''    let mut tokens = [0usize; SEQ];
    let prompt_tokens = [11, 13, 30, 1, 19, 0];
    for (i, &t) in prompt_tokens.iter().enumerate() {
        tokens[i] = t;
    }
    let prompt_len = 6;'''

assert old in content, "Failed old"
content = content.replace(old, new, 1)

old2 = '''    let last_token_idx = prompt.len() - 1;'''
new2 = '''    let last_token_idx = prompt_len - 1;'''
assert old2 in content, "Failed old2"
content = content.replace(old2, new2, 1)

open("examples/compare_rust.rs", "w").write(content)
print("patched")

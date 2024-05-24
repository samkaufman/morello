pub fn clang_path() -> Option<String> {
    match std::env::var("CLANG") {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}

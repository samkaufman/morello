pub fn clang_path() -> Option<String> {
    std::env::var("CLANG").ok()
}

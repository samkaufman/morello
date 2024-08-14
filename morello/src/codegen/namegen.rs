pub struct NameGenerator {
    names_generated: usize,
}

impl NameGenerator {
    pub fn new() -> Self {
        NameGenerator { names_generated: 0 }
    }

    pub fn fresh_name(&mut self) -> String {
        let new_name = format!("n{:03}", self.names_generated);
        self.names_generated += 1;
        new_name
    }
}

impl Default for NameGenerator {
    fn default() -> Self {
        Self::new()
    }
}

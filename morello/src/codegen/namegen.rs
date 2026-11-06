use std::iter;

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

    pub fn fresh_names(&mut self, count: usize) -> Vec<String> {
        iter::repeat_with(|| self.fresh_name())
            .take(count)
            .collect()
    }
}

impl Default for NameGenerator {
    fn default() -> Self {
        Self::new()
    }
}

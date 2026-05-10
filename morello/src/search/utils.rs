/// Tracks which working-set tasks have been visited during the current
/// [super::blocksearch::synthesize] iteration.
pub struct StageVisitSet {
    generation: u32,
    marks: Vec<u32>,
}

impl StageVisitSet {
    pub fn new(initial_len: usize) -> Self {
        Self {
            generation: 1,
            marks: vec![0; initial_len],
        }
    }

    pub fn reset_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    pub fn insert(&mut self, working_set_idx: usize) -> bool {
        if working_set_idx >= self.marks.len() {
            self.marks.resize(working_set_idx + 1, 0);
        }

        let mark = &mut self.marks[working_set_idx];
        if *mark == self.generation {
            false
        } else {
            *mark = self.generation;
            true
        }
    }
}

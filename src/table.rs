use crate::common::Problem;
use crate::cost::Cost;
use crate::imp::ImplNode;
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::spec::Spec;
use crate::target::Target;
use std::collections::HashMap;

// An extremely simple cache, mapping Specs and memory bounds to implementations.
pub struct Database<Tgt: Target> {
    grouped_entries: HashMap<Spec<Tgt>, Entry<Tgt>>,
}

#[derive(Debug)]
struct Entry<Tgt: Target> {
    ranges: Vec<(MemVec, MemVec)>,
    values: Vec<Vec<(ImplNode<Tgt>, Cost)>>,
}

impl<Tgt: Target> Database<Tgt> {
    pub fn new() -> Self {
        Database {
            grouped_entries: HashMap::new(),
        }
    }

    pub fn get(&self, query: &Problem<Tgt>) -> Option<&Vec<(ImplNode<Tgt>, Cost)>> {
        match (self.grouped_entries.get(&query.0), &query.1) {
            (Some(entry), MemoryLimits::Standard(query_lims)) => {
                for (i, (lims, peaks)) in entry.ranges.iter().enumerate() {
                    if query_lims
                        .iter()
                        .zip(lims)
                        .zip(peaks)
                        .all(|((q, l), p)| l >= *q && *q >= p)
                    {
                        return Some(&entry.values[i]);
                    }
                }
                None
            }
            (None, _) => None,
        }
    }

    pub fn put(&mut self, problem: Problem<Tgt>, impls: Vec<(ImplNode<Tgt>, Cost)>) {
        // self.entries.insert(problem, impls);

        let orig_problem = problem.clone(); // Save for debug_assert_eq! postcondition.
        let orig_impls = impls.clone();

        // TODO: How to treat non-powers of two memory bounds?
        match problem.1 {
            MemoryLimits::Standard(lims) => {
                // for intermediate_memory in lims
                //     .iter()
                //     .zip(&impls)
                //     .map(|(l, p)| p.1.main..l + 1) // TODO: next power of two
                //     .multi_cartesian_product()
                // // TODO: Product of powers of two
                // {
                //     self.entries.insert(
                //         Problem(
                //             problem.0.clone(),
                //             MemoryLimits::Standard(intermediate_memory.into()),
                //         ),
                //         impls.clone(),
                //     );
                // }

                assert!(impls.len() < 2);
                let existing: &mut Entry<Tgt> = self.grouped_entries.entry(problem.0).or_default();
                if impls.len() == 0 {
                    existing.ranges.push((lims.clone(), MemVec::zero::<Tgt>()));
                } else {
                    existing
                        .ranges
                        .push((lims.clone(), impls[0].1.peaks.clone()));
                }
                existing.values.push(impls);
            }
        }

        debug_assert!(
            self.get(&orig_problem).is_some(),
            "Original problem {:?} was missing after put.  The Spec entry contained: {:?}",
            orig_problem,
            self.grouped_entries.get(&orig_problem.0)
        );
        debug_assert_eq!(
            self.get(&orig_problem),
            Some(&orig_impls),
            "Original problem {:?} was incorrect after put. The Spec entry contained: {:?}",
            orig_problem,
            self.grouped_entries.get(&orig_problem.0)
        );
    }
}

impl<Tgt: Target> Default for Entry<Tgt> {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            values: Default::default(),
        }
    }
}

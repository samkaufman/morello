use crate::common::Problem;
use crate::cost::Cost;
use crate::imp::ImplNode;
use crate::table::{Database, DatabaseIOStore};
use crate::target::Target;

struct ImplReducer<Tgt: Target> {
    results: Vec<(ImplNode<Tgt>, Cost)>,
    top_k: usize,
}

// TODO: Would be better to return a reference to the database, not a clone.
/// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<Tgt: Target, S: DatabaseIOStore<Tgt>>(
    db: &mut Database<Tgt, S>,
    goal: &Problem<Tgt>,
    top_k: usize,
) -> (Vec<(ImplNode<Tgt>, Cost)>, u64, u64) {
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // First, check if the problem is already in the database.
    if let Some(stored) = db.get(goal) {
        return (stored.clone(), 1, 0);
    }

    let mut hits = 0u64;
    let mut misses = 1u64;

    // Enumerate expansions, computing their costs from their childrens' costs.
    let mut reducer = ImplReducer::new(top_k);
    for (expanded_node, child_mem_bounds) in goal.expansions() {
        let mut child_sub_costs = Vec::with_capacity(child_mem_bounds.len());

        // Recurse into children.
        let mut unsat = false;
        let child_specs = expanded_node.child_specs(&goal.0);
        debug_assert_eq!(
            child_specs.len(),
            child_mem_bounds.len(),
            "Got {} child Specs but {} memory bounds for {:?}",
            child_specs.len(),
            child_mem_bounds.len(),
            expanded_node
        );
        for (child, mlims) in child_specs.iter().zip(child_mem_bounds) {
            let (child_result, subhits, submisses) =
                top_down(db, &Problem(child.clone(), mlims), top_k);
            hits += subhits;
            misses += submisses;
            if child_result.is_empty() {
                unsat = true;
                break;
            }
            child_sub_costs.push(child_result[0].1.clone()); // TODO: Should move
        }
        if unsat {
            continue;
        }

        let cost = Cost::from_child_costs(&goal.0, &expanded_node, child_sub_costs);
        reducer.insert(expanded_node, cost);
    }

    // Save to memo. table and return.
    let final_result = reducer.finalize();
    db.put(goal.clone(), final_result.clone());
    (final_result, hits, misses)
}

impl<Tgt: Target> ImplReducer<Tgt> {
    fn new(top_k: usize) -> Self {
        ImplReducer {
            results: vec![],
            top_k,
        }
    }

    fn insert(&mut self, new_impl: ImplNode<Tgt>, cost: Cost) {
        self.results.push((new_impl, cost));
    }

    fn finalize(&self) -> Vec<(ImplNode<Tgt>, Cost)> {
        // Using sorted here for stability.
        let mut sorted_results = self.results.clone();
        sorted_results.sort_by_key(|x| x.1.clone());
        sorted_results.truncate(self.top_k);
        sorted_results
        // return heapq.nsmallest(top_k, results, key=lambda x: x[1])
    }
}

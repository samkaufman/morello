use std::hash::Hash;

#[cfg(debug_assertions)]
use std::collections::HashSet;

pub trait Problem {
    /// A node in the search DAG.
    type Node: Clone + Eq + Hash;
    /// Identifier of a dependency request.
    ///
    /// Provided by [Problem::next_request_batch] and returned with [Problem::resolve_request].
    type Request;
    type Task;
    type Value: Clone;

    fn start(&self, node: &Self::Node) -> Self::Task;

    fn next_request_batch(&self, task: &mut Self::Task)
        -> Option<Vec<(Self::Node, Self::Request)>>;

    fn resolve_request(
        &self,
        task: &mut Self::Task,
        request: Self::Request,
        child_value: Self::Value,
    );

    fn finish(&self, task: Self::Task) -> Self::Value;
}

pub trait MemoDatabase<N, V> {
    fn get(&mut self, node: &N) -> Option<V>;
    fn put(&mut self, node: N, value: V);
}

struct SearchState<Pb, M>
where
    Pb: Problem,
{
    problem: Pb,
    memo: M,
    /// A set of nodes for identifying cycles in debug builds .
    #[cfg(debug_assertions)]
    active: HashSet<Pb::Node>,
}

impl<Pb, M> SearchState<Pb, M>
where
    Pb: Problem,
    M: MemoDatabase<Pb::Node, Pb::Value>,
{
    fn solve_node(&mut self, node: Pb::Node) -> Pb::Value {
        #[cfg(debug_assertions)]
        debug_assert!(!self.active.contains(&node), "cyclic dependency detected");

        if let Some(value) = self.memo.get(&node) {
            return value;
        }

        #[cfg(debug_assertions)]
        debug_assert!(self.active.insert(node.clone()));

        let mut task = self.problem.start(&node);
        while let Some(batch) = self.problem.next_request_batch(&mut task) {
            let mut child_values = Vec::with_capacity(batch.len());
            for (child, request) in batch {
                let child_value = self.solve_node(child);
                child_values.push((request, child_value));
            }

            for (request, child_value) in child_values {
                self.problem
                    .resolve_request(&mut task, request, child_value);
            }
        }

        let result = self.problem.finish(task);

        #[cfg(debug_assertions)]
        debug_assert!(self.active.remove(&node));

        let value = result.clone();
        self.memo.put(node, result);
        value
    }
}

pub fn solve<Pb, M>(problem: Pb, memo: M, root: Pb::Node) -> Pb::Value
where
    Pb: Problem,
    M: MemoDatabase<Pb::Node, Pb::Value>,
{
    let mut s = SearchState {
        problem,
        memo,
        #[cfg(debug_assertions)]
        active: HashSet::new(),
    };
    s.solve_node(root)
}

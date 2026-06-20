use crate::db::{ActionCostVec, FilesDatabase, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::spec::Spec;
use crate::target::Target;

use super::algo::{MemoDatabase, Problem};
use super::{RequestId, SpecTask};

#[derive(Clone, Default)]
pub struct SpecProblem<Tgt: Target>(std::marker::PhantomData<Tgt>); // TODO: Rename to OptProblem

impl<Tgt> Problem for SpecProblem<Tgt>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    type Node = Spec<Tgt>;
    type Request = RequestId;
    type Task = SpecTask<Tgt>;
    type Value = ActionCostVec;

    fn start(&self, node: &Self::Node) -> Self::Task {
        debug_assert!(node.is_canonical());
        SpecTask::start(node)
    }

    fn next_request_batch(
        &self,
        task: &mut Self::Task,
    ) -> Option<Vec<(Self::Node, Self::Request)>> {
        task.next_request_batch().map(|batch| {
            batch
                .map(|(subspec, request_id)| {
                    debug_assert!(subspec.is_canonical());
                    (subspec, request_id)
                })
                .collect()
        })
    }

    fn resolve_request(
        &self,
        task: &mut Self::Task,
        request: Self::Request,
        child_value: Self::Value,
    ) -> bool {
        if task.is_running() {
            let cost = child_value.0.into_iter().next().map(|(_, cost)| cost);
            task.resolve_request(request, cost);
        }
        task.is_running()
    }

    fn finish(&self, task: Self::Task) -> Self::Value {
        task.into_result()
            .expect("task should be complete once all children are solved")
            .0
    }
}

impl<Tgt> MemoDatabase<Spec<Tgt>, ActionCostVec> for &FilesDatabase
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    fn get(&mut self, node: &Spec<Tgt>) -> Option<ActionCostVec> {
        debug_assert!(node.is_canonical());
        match self.get_with_preference_canon(node) {
            GetPreference::Hit(result) => Some(result),
            GetPreference::Miss(_) => None,
        }
    }

    fn put(&mut self, node: Spec<Tgt>, value: ActionCostVec) {
        self.put_canon(&node, value.0);
    }
}

use crate::cost::Cost;
use crate::db::ActionNum;
use itertools::Itertools;
use std::collections::BTreeSet;

// TODO: Make this private once #[bench] gets stable.
#[doc(hidden)]
#[derive(Debug)]
pub struct ImplReducer {
    results: ImplReducerResults,
    top_k: usize,
    preferences: Vec<ActionNum>,
}

#[derive(Debug)]
enum ImplReducerResults {
    One(Option<(Cost, ActionNum)>),
    Many(BTreeSet<(Cost, ActionNum)>),
}

impl ImplReducer {
    pub fn new(top_k: usize, preferences: Vec<ActionNum>) -> Self {
        debug_assert!(preferences.len() <= top_k);
        debug_assert!(
            preferences.iter().all_unique(),
            "Preferences should not contain duplicates"
        );

        ImplReducer {
            results: if top_k == 1 {
                ImplReducerResults::One(None)
            } else {
                ImplReducerResults::Many(BTreeSet::new())
            },
            top_k,
            preferences,
        }
    }

    pub fn insert(&mut self, new_action_num: ActionNum, new_cost: Cost) {
        let new_action = (new_cost, new_action_num);
        match &mut self.results {
            ImplReducerResults::One(None) => {
                self.results = ImplReducerResults::One(Some(new_action));
            }
            ImplReducerResults::One(Some(action)) if *action > new_action => {
                self.results = ImplReducerResults::One(Some(new_action));
            }
            ImplReducerResults::Many(ref mut actions) => {
                if actions.len() < self.top_k {
                    // We have not yet filled the top_k, so just insert.
                    actions.insert(new_action);
                } else if actions.iter().any(|(cost, _)| *cost == new_action.0) {
                    debug_assert_eq!(actions.len(), self.top_k);

                    // We have filled the top_k and found the same cost in results, so
                    //   replace something if it improves preference count, and do
                    //   nothing if not.
                    if let Some((_, action)) = actions
                        .iter()
                        .enumerate()
                        // Since we know that results is sorted by Cost, this filter
                        //   only takes contiguous elements with the same cost.
                        .filter(|&(i, (cost, _))| {
                            i < self.preferences.len() && *cost == new_action.0
                        })
                        .find(|&(i, _)| self.preferences[i] == new_action.1)
                    {
                        actions.remove(&action.clone());
                        actions.insert(new_action);
                    }
                } else {
                    debug_assert_eq!(actions.len(), self.top_k);

                    // We have filled the top_k, but there is no same cost in results,
                    //   so replace the last element if it is worse than the new one.
                    actions.insert(new_action);
                    actions.pop_last();
                }

                debug_assert!(actions.iter().tuple_windows().all(|(a, b)| a.0 <= b.0));
                debug_assert!(actions.len() <= self.top_k);
                debug_assert!(actions.iter().map(|(_, a)| a).all_unique());
            }
            _ => {}
        }
    }

    pub fn finalize(self) -> Vec<(ActionNum, Cost)> {
        match self.results {
            ImplReducerResults::One(None) => vec![],
            ImplReducerResults::One(Some((cost, action_num))) => vec![(action_num, cost)],
            ImplReducerResults::Many(actions) => actions
                .into_iter()
                .map(|(cost, action_num)| (action_num, cost))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memorylimits::MemVec, target::Avx2Target};
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn test_implreducer_no_actions() {
        let top_k = 1;
        let preferences = vec![];
        let reducer = ImplReducer::new(top_k, preferences);

        let expected: Vec<_> = vec![];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_exactly_one_action() {
        let top_k = 1;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost1.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_cost() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);
        let cost3 = create_simple_cost(3);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost3.clone());
        reducer.insert(2, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1), (2, cost2), (1, cost3)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_action_num() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost1.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (2, cost1.clone())];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_cost_then_action_num() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost2.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(1, cost1.clone()), (2, cost1.clone()), (0, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement() {
        let top_k = 3;
        let preferences = vec![0, 2, 3];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost1.clone());
        reducer.insert(2, cost1.clone());
        reducer.insert(3, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (3, cost1)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement_and_sort_by_cost() {
        let top_k = 3;
        let preferences = vec![0, 2, 3];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost2.clone());
        reducer.insert(1, cost2.clone());
        reducer.insert(2, cost2.clone());
        reducer.insert(3, cost1.clone());

        let expected: Vec<_> = vec![(3, cost1.clone()), (0, cost2.clone()), (1, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement_and_sort_by_cost_then_action_num() {
        let top_k = 3;
        let preferences = vec![3, u16::MAX, 0];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost2.clone());
        reducer.insert(2, cost1.clone());
        // 0, 2, 1

        reducer.insert(3, cost1.clone());
        // 3, 2, 1 -> 2, 3, 1

        let expected: Vec<_> = vec![(2, cost1.clone()), (3, cost1.clone()), (1, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_cost_replacement() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);
        let cost3 = create_simple_cost(3);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost3.clone());
        reducer.insert(2, cost3.clone());
        reducer.insert(3, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1), (3, cost2), (1, cost3)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_no_cost_replacement() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost1.clone());
        reducer.insert(2, cost1.clone());
        reducer.insert(3, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (2, cost1.clone())];
        assert_eq!(reducer.finalize(), expected, "no replacement should occur");
    }

    proptest! {
        // applies to this `proptest!` block
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn test_implreducer_can_sort_any_top_k_actions(
            (top_k, mut action_costs) in arb_top_k_and_action_costs()
        ) {
            let preferences = vec![];
            let mut reducer = ImplReducer::new(top_k, preferences);

            for (cost, action_num) in &action_costs {
                reducer.insert(*action_num, cost.clone());
            }

            let finalized = reducer.finalize();
            action_costs.sort();
            assert_eq!(finalized.len(), action_costs.len());

            for (reduced, original) in finalized.into_iter().zip(action_costs.into_iter().map(|(action_num, cost)| (cost, action_num))) {
                assert_eq!(reduced, original);
            }
        }
    }

    prop_compose! {
        fn arb_top_k_and_action_costs()(top_k in 2..128usize)
        (
            top_k in Just(top_k),
            action_indices in arb_action_indices(top_k),
            costs in arb_costs(top_k)
        ) -> (usize, Vec<(Cost, ActionNum)>) {
            (top_k, costs.into_iter().zip(action_indices).collect())
        }
    }

    fn arb_action_indices(top_k: usize) -> impl Strategy<Value = HashSet<ActionNum>> {
        prop::collection::hash_set(any::<ActionNum>(), 1..top_k)
    }

    fn arb_costs(top_k: usize) -> impl Strategy<Value = Vec<Cost>> {
        prop::collection::vec(any::<Cost>(), 1..top_k)
    }

    fn create_simple_cost(main: u32) -> Cost {
        Cost {
            main,
            peaks: MemVec::zero::<Avx2Target>(),
            depth: 0,
        }
    }
}

use crate::common::{DimSize, Dtype};
use crate::db::FilesDatabase;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::functions::FunctionApp;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::scheduling::{Action, ApplyError, TileOut};
use crate::search::top_down;
use crate::spec::Spec;
use crate::target::Target;
use crate::views::ViewE;
use std::iter;
use std::num::NonZeroUsize;

/// A trait extending [ImplNode]s and [Spec]s with methods for more conveniently applying [Action]s.
///
/// These methods are intended to be used for manual scheduling by developers. Other clients should
/// probably apply [Action]s directly.
pub trait SchedulingSugar<Tgt: Target> {
    fn tile_out(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn tile_out_parallel(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn split(&self, k: u32) -> ImplNode<Tgt>;
    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn cast(
        &self,
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn to_accum(&self) -> ImplNode<Tgt>;
    fn to_softmax_parts(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: Layout,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn to_max_and_denominator(&self) -> ImplNode<Tgt>;
    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn spatial_split(&self) -> ImplNode<Tgt>;
    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt>;
    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt>;
    fn synthesize(&self, db: &FilesDatabase, jobs: Option<NonZeroUsize>) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;
}

pub trait Subschedule<Tgt: Target> {
    /// Apply a function to schedule a nested sub-Spec.
    fn subschedule(&self, path: &[usize], f: impl Fn(&Spec<Tgt>) -> ImplNode<Tgt>)
        -> ImplNode<Tgt>;
}

impl<Tgt: Target> SchedulingSugar<Tgt> for Spec<Tgt> {
    fn tile_out(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        let action = Action::TileOut(TileOut::MultiLoop {
            output_shape: output_shape
                .iter()
                .map(|&d| DimSize::new(d).unwrap())
                .collect(),
            parallel: false,
        });
        apply_unwrap(self, action)
    }

    fn tile_out_parallel(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        let action = Action::TileOut(TileOut::MultiLoop {
            output_shape: output_shape
                .iter()
                .map(|&d| DimSize::new(d).unwrap())
                .collect(),
            parallel: true,
        });
        apply_unwrap(self, action)
    }

    fn split(&self, k: u32) -> ImplNode<Tgt> {
        let action = Action::Split {
            k: DimSize::new(k).unwrap(),
        };
        apply_unwrap(self, action)
    }

    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let destination_dtype = self.0.parameters()[usize::from(source_idx)].dtype();
        let action = Action::Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout,
            destination_vector_size,
        };
        apply_unwrap(self, action)
    }

    fn cast(
        &self,
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let action = Action::Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout,
            destination_vector_size,
        };
        apply_unwrap(self, action)
    }

    fn to_accum(&self) -> ImplNode<Tgt> {
        let action = Action::ToAccum;
        apply_unwrap(self, action)
    }

    fn to_softmax_parts(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: Layout,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_unwrap(
            self,
            Action::ToSoftmaxParts {
                max_level,
                max_layout,
                max_vector_size,
                denominator_level,
                denominator_layout,
                denominator_vector_size,
            },
        )
    }

    fn to_max_and_denominator(&self) -> ImplNode<Tgt> {
        apply_unwrap(self, Action::ToMaxAndDenominator)
    }

    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_unwrap(
            self,
            Action::Bufferize {
                index,
                level,
                layout,
                vector_size,
            },
        )
    }

    fn spatial_split(&self) -> ImplNode<Tgt> {
        let action = Action::SpatialSplit;
        apply_unwrap(self, action)
    }

    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        let action = Action::Select(kernel.into(), false);
        apply_unwrap(self, action)
    }

    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        let action = Action::Select(kernel.into(), true);
        apply_unwrap(self, action)
    }

    fn synthesize(&self, db: &FilesDatabase, jobs: Option<NonZeroUsize>) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        top_down(db, self, 1, jobs);
        match db.get_impl(self).unwrap().first() {
            Some(imp) => imp.clone(),
            None => panic!("No Impl exists for {self}"),
        }
    }
}

impl<Tgt: Target> SchedulingSugar<Tgt> for ImplNode<Tgt> {
    fn tile_out(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.tile_out(output_shape))
    }

    fn tile_out_parallel(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.tile_out_parallel(output_shape))
    }

    fn split(&self, k: u32) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.split(k))
    }

    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.move_param(
                source_idx,
                destination_level,
                destination_layout,
                destination_vector_size,
            )
        })
    }

    fn cast(
        &self,
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.cast(
                source_idx,
                destination_dtype,
                destination_level,
                destination_layout,
                destination_vector_size,
            )
        })
    }

    fn to_accum(&self) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.to_accum())
    }

    fn to_softmax_parts(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: Layout,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.to_softmax_parts(
                max_level,
                max_layout,
                max_vector_size,
                denominator_level,
                denominator_layout,
                denominator_vector_size,
            )
        })
    }

    fn to_max_and_denominator(&self) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.to_max_and_denominator())
    }

    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.bufferize(index, level, layout, vector_size)
        })
    }

    fn spatial_split(&self) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.spatial_split())
    }

    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.select(kernel))
    }

    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.force_select(kernel))
    }

    fn synthesize(&self, db: &FilesDatabase, jobs: Option<NonZeroUsize>) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        apply_to_leaf_spec(self, |spec| spec.synthesize(db, jobs))
    }
}

impl<Tgt: Target> Subschedule<Tgt> for ImplNode<Tgt> {
    fn subschedule(
        &self,
        path: &[usize],
        f: impl Fn(&Spec<Tgt>) -> ImplNode<Tgt>,
    ) -> ImplNode<Tgt> {
        fn inner<Tgt: Target>(
            this: &ImplNode<Tgt>,
            path: &[usize],
            f: &impl Fn(&Spec<Tgt>) -> ImplNode<Tgt>,
        ) -> ImplNode<Tgt> {
            let children = this.children();
            if children.is_empty() {
                match this {
                    ImplNode::SpecApp(spec_app) => specapp_apply(spec_app, f).into(),
                    _ => panic!("subschedule path chose non-Spec leaf: {:?}", this),
                }
            } else if children.len() == 1 {
                this.replace_children(iter::once(inner(&children[0], path, f)))
            } else if path[0] >= children.len() {
                panic!(
                    "subschedule path referenced child {} but only {} children",
                    path[0],
                    children.len()
                );
            } else {
                this.replace_children(children.iter().enumerate().map(|(i, child)| {
                    if i == path[0] {
                        inner(child, &path[1..], f)
                    } else {
                        child.clone()
                    }
                }))
            }
        }

        inner(self, path, &f)
    }
}

fn apply_to_leaf_spec<Tgt, F>(node: &ImplNode<Tgt>, f: F) -> ImplNode<Tgt>
where
    Tgt: Target,
    F: FnOnce(&Spec<Tgt>) -> ImplNode<Tgt>,
{
    match node {
        ImplNode::SpecApp(app) => specapp_apply(app, f).into(),
        _ => match &node.children() {
            [] => panic!("Not a Spec application and no children."),
            [child] => node.replace_children(iter::once(apply_to_leaf_spec(child, f))),
            children => {
                if let Some(default_child_idx) = node.default_child() {
                    let replaced_child = apply_to_leaf_spec(&children[default_child_idx], f);
                    node.replace_children(
                        children[..default_child_idx]
                            .iter()
                            .chain(iter::once(&replaced_child))
                            .chain(&children[(default_child_idx + 1)..])
                            .cloned(),
                    )
                } else {
                    panic!("Ambiguous choice of child. Use `subschedule`.")
                }
            }
        },
    }
}

fn apply_unwrap<Tgt: Target>(spec: &Spec<Tgt>, action: Action<Tgt>) -> ImplNode<Tgt> {
    match action.apply(spec) {
        Ok(result) => result,
        Err(ApplyError::NotApplicable(reason)) => {
            panic!("Action {action:?} is not defined for {spec}: {reason}")
        }
        Err(e) => panic!("{e}"),
    }
}

fn specapp_apply<Tgt, F>(app: &SpecApp<ViewE<Tgt>>, f: F) -> FunctionApp<Tgt>
where
    Tgt: Target,
    F: FnOnce(&Spec<Tgt>) -> ImplNode<Tgt>,
{
    FunctionApp {
        body: Box::new(f(&app.0)),
        parameters: app.1.clone(),
        spec: Some(app.0.clone()),
    }
}

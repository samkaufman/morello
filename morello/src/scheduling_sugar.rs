use crate::common::{DimSize, Dtype};
use crate::db::FilesDatabase;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::functions::FunctionApp;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::{Layout, LayoutBuilder};
use crate::scheduling::broadcast_first::BroadcastFirst;
use crate::scheduling::bufferize::Bufferize;
use crate::scheduling::moves::Move;
use crate::scheduling::select::Select;
use crate::scheduling::spatial_split::SpatialSplit;
use crate::scheduling::tiling::{Split, TileOut};
use crate::scheduling::to_accum::ToAccum;
use crate::scheduling::to_max_and_denom::ToMaxAndDenominator;
use crate::scheduling::to_max_and_unscaled::ToMaxAndUnscaled;
use crate::scheduling::to_softmax_parts::{ToSoftmaxParts, ToSoftmaxPartsRecompute};
use crate::scheduling::ActionT as _;
use crate::scheduling::{Action, ApplyError};
use crate::search::top_down;
use crate::spec::{LogicalSpec, Spec};
use crate::target::Target;
use crate::views::ViewE;
use std::iter;

/// A trait extending [ImplNode]s and [Spec]s with methods for more conveniently applying [Action]s.
///
/// These methods are intended to be used for manual scheduling by developers. Other clients should
/// probably apply [Action]s directly.
pub trait SchedulingSugar<Tgt: Target> {
    fn tile_out(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn tile_out_parallel(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn split(&self, k: u32) -> ImplNode<Tgt>;
    fn move_param(&self, source_idx: u8, destination_level: Tgt::Level) -> ImplNode<Tgt>;
    fn move_vrf(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_vector_size: DimSize,
    ) -> ImplNode<Tgt>;
    fn move_relayout(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: impl LayoutBuilder,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn cast(
        &self,
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: impl LayoutBuilder,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn to_accum(&self) -> ImplNode<Tgt>;
    fn to_softmax_parts(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        exps_level: Tgt::Level,
        exps_layout: Layout,
        exps_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn to_softmax_parts_recompute(
        &self,
        max_level: Tgt::Level,
        max_layout: impl LayoutBuilder,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: impl LayoutBuilder,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn to_max_and_denominator(&self) -> ImplNode<Tgt>;
    fn to_max_and_unscaled(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: impl LayoutBuilder,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn spatial_split(&self) -> ImplNode<Tgt>;
    fn broadcast_first(
        &self,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt>;
    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt>;
    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt>;
    fn synthesize(&self, db: &FilesDatabase) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;
    fn synthesize_all(&self, db: &FilesDatabase) -> ImplNode<Tgt>
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
        let action = Action::Split(Split {
            k: DimSize::new(k).unwrap(),
        });
        apply_unwrap(self, action)
    }

    fn move_param(&self, source_idx: u8, destination_level: Tgt::Level) -> ImplNode<Tgt> {
        let dest = self.0.parameter(usize::from(source_idx));
        let destination_dtype = dest.dtype();
        let action = Action::Move(Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout: dest.layout().clone(),
            destination_vector_size: None,
        });
        apply_unwrap(self, action)
    }

    fn move_vrf(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_vector_size: DimSize,
    ) -> ImplNode<Tgt> {
        let dest = self.0.parameter(usize::from(source_idx));
        let destination_dtype = dest.dtype();
        let action = Action::Move(Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout: dest.layout().clone(),
            destination_vector_size: Some(destination_vector_size),
        });
        apply_unwrap(self, action)
    }

    fn move_relayout(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: impl LayoutBuilder,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let dest = self.0.parameters().swap_remove(usize::from(source_idx));
        let destination_dtype = dest.dtype();
        let action = Action::Move(Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout: destination_layout.build(dest.shape()),
            destination_vector_size,
        });
        apply_unwrap(self, action)
    }

    fn cast(
        &self,
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: impl LayoutBuilder,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let dest_shape = self.0.parameter_shape(source_idx.into());
        let action = Action::Move(Move {
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout: destination_layout.build(&dest_shape),
            destination_vector_size,
        });
        apply_unwrap(self, action)
    }

    fn to_accum(&self) -> ImplNode<Tgt> {
        let action = Action::ToAccum(ToAccum::default());
        apply_unwrap(self, action)
    }

    fn to_softmax_parts(
        &self,
        denominator_level: <Tgt as Target>::Level,
        denominator_layout: Layout,
        denominator_vector_size: Option<DimSize>,
        exps_level: <Tgt as Target>::Level,
        exps_layout: Layout,
        exps_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_unwrap(
            self,
            Action::ToSoftmaxParts(ToSoftmaxParts {
                denominator_level,
                denominator_layout,
                denominator_vector_size,
                exps_level,
                exps_layout,
                exps_vector_size,
            }),
        )
    }

    fn to_softmax_parts_recompute(
        &self,
        max_level: Tgt::Level,
        max_layout: impl LayoutBuilder,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: impl LayoutBuilder,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let first_parameter_shape = self.0.parameter_shape(0);
        let max_layout = max_layout.build(&first_parameter_shape);
        let denominator_layout = denominator_layout.build(&first_parameter_shape);
        apply_unwrap(
            self,
            Action::ToSoftmaxPartsRecompute(ToSoftmaxPartsRecompute {
                max_level,
                max_layout,
                max_vector_size,
                denominator_level,
                denominator_layout,
                denominator_vector_size,
            }),
        )
    }

    fn to_max_and_denominator(&self) -> ImplNode<Tgt> {
        apply_unwrap(
            self,
            Action::ToMaxAndDenominator(ToMaxAndDenominator::default()),
        )
    }

    fn to_max_and_unscaled(
        &self,
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_unwrap(
            self,
            Action::ToMaxAndUnscaled(ToMaxAndUnscaled {
                max_level,
                max_layout,
                max_vector_size,
            }),
        )
    }

    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: impl LayoutBuilder,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let LogicalSpec::Compose { components, .. } = &self.0 else {
            panic!("Not a Compose");
        };
        let consumer = &components[index];
        apply_unwrap(
            self,
            Action::Bufferize(Bufferize {
                index,
                level,
                layout: layout.build(&consumer.parameter_shape(0)),
                vector_size,
            }),
        )
    }

    fn spatial_split(&self) -> ImplNode<Tgt> {
        apply_unwrap(self, Action::SpatialSplit(SpatialSplit::default()))
    }

    fn broadcast_first(
        &self,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        let action = Action::BroadcastFirst(BroadcastFirst {
            broadcast_level: level,
            broadcast_layout: layout,
            broadcast_vector_size: vector_size,
        });
        apply_unwrap(self, action)
    }

    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        let action = Action::Select(Select(kernel.into(), false));
        apply_unwrap(self, action)
    }

    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        let action = Action::Select(Select(kernel.into(), true));
        apply_unwrap(self, action)
    }

    fn synthesize(&self, db: &FilesDatabase) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        top_down(db, self, 1);
        match db.get_impl(self).unwrap().first() {
            Some(imp) => imp.clone(),
            None => panic!("No Impl exists for {self}"),
        }
    }

    fn synthesize_all(&self, db: &FilesDatabase) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        self.synthesize(db)
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

    fn move_param(&self, source_idx: u8, destination_level: Tgt::Level) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.move_param(source_idx, destination_level))
    }

    fn move_vrf(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_vector_size: DimSize,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.move_vrf(source_idx, destination_level, destination_vector_size)
        })
    }

    fn move_relayout(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: impl LayoutBuilder,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.move_relayout(
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
        destination_layout: impl LayoutBuilder,
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
        max_level: <Tgt as Target>::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        exps_level: <Tgt as Target>::Level,
        exps_layout: Layout,
        exps_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.to_softmax_parts(
                max_level,
                max_layout,
                max_vector_size,
                exps_level,
                exps_layout,
                exps_vector_size,
            )
        })
    }

    fn to_softmax_parts_recompute(
        &self,
        max_level: Tgt::Level,
        max_layout: impl LayoutBuilder,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: impl LayoutBuilder,
        denominator_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.to_softmax_parts_recompute(
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

    fn to_max_and_unscaled(
        &self,
        max_level: <Tgt as Target>::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.to_max_and_unscaled(max_level, max_layout, max_vector_size)
        })
    }

    fn bufferize(
        &self,
        index: usize,
        level: Tgt::Level,
        layout: impl LayoutBuilder,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.bufferize(index, level, layout, vector_size)
        })
    }

    fn spatial_split(&self) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.spatial_split())
    }

    fn broadcast_first(
        &self,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| {
            spec.broadcast_first(level, layout, vector_size)
        })
    }

    fn select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.select(kernel))
    }

    fn force_select<T: Into<Tgt::Kernel>>(&self, kernel: T) -> ImplNode<Tgt> {
        apply_to_leaf_spec(self, |spec| spec.force_select(kernel))
    }

    fn synthesize(&self, db: &FilesDatabase) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        apply_to_leaf_spec(self, |spec| spec.synthesize(db))
    }

    fn synthesize_all(&self, db: &FilesDatabase) -> ImplNode<Tgt>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        apply_to_leaves(self, &|spec| spec.synthesize(db))
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
                    _ => panic!("subschedule path chose non-Spec leaf: {this:?}"),
                }
            } else if children.len() == 1 {
                this.replace_children(iter::once(inner(&children[0], path, f)))
            } else if path.is_empty() {
                panic!("subschedule path is too short");
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

fn apply_to_leaves<Tgt, F>(node: &ImplNode<Tgt>, f: &F) -> ImplNode<Tgt>
where
    Tgt: Target,
    F: Fn(&Spec<Tgt>) -> ImplNode<Tgt>,
{
    match node {
        ImplNode::SpecApp(app) => specapp_apply(app, f).into(),
        _ => node.replace_children(
            node.children()
                .iter()
                .map(|child| apply_to_leaves(child, f)),
        ),
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

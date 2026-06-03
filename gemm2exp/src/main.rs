use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::imp::functions::FunctionApp;
use morello::imp::subspecs::SpecApp;
use morello::imp::{Impl, ImplNode};
use morello::layout::row_major;
use morello::scheduling::bufferize::Bufferize;
use morello::scheduling::tiling::TileOut;
use morello::scheduling::{ActionT, ApplyError};
use morello::search::top_down_many_impls;
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{Avx2Target, CpuMemory::GL, Memory, Target};
use morello::tensorspec::TensorSpecAux;

use clap::Parser;
use itertools::Itertools;
use std::error::Error;
use std::io;
use std::path;

const BATCH: u32 = 1;
const N: u32 = 32;
const K: u32 = 128;
const KO: u32 = 128;
const DTYPE: Dtype = Dtype::Float32;
const ITERS: u32 = 10;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    db: Option<path::PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let db = morello::db::FilesDatabase::new::<Avx2Target>(
        args.db.as_deref(),
        morello::db::TileScale::PowerOfTwo,
        1,
        10_000,
        1,
    );
    let spec = make_spec();
    let mut writer = csv::Writer::from_writer(io::stdout());

    writer.write_record([
        "implementation",
        "tile_output_shape",
        "bufferize_index",
        "bufferize_memory",
        "bufferize_layout",
        "bufferize_vector_size",
        "analytical_cost",
        "runtime_seconds",
    ])?;

    let mut implementation_id = 0usize;

    for (b, n) in (1..=BATCH).zip(1..=N) {
        let tile_output_shape = shape![b, n, n];
        let tile = TileOut::MultiLoop {
            output_shape: tile_output_shape.clone(),
            parallel: false,
        };
        let Ok(tiled) = tile.apply(&spec) else {
            continue;
        };
        let Ok(ImplNode::SpecApp(SpecApp(tiled_spec, _))) = tiled.children().iter().exactly_one()
        else {
            eprintln!(
                "skipping tile {}: could not find default Spec leaf",
                shape_str(&tile_output_shape)
            );
            continue;
        };

        for bufferize in bufferize_params(tiled_spec) {
            let Ok(scheduled) = apply_bufferize_to_default_leaf(tiled.clone(), &bufferize) else {
                eprintln!(
                    "skipping tile {}, bufferize {:?}: action did not apply",
                    shape_str(&tile_output_shape),
                    bufferize,
                );
                continue;
            };

            let Some(implementation) = synthesize_all_impls(scheduled, &db) else {
                eprintln!(
                    "skipping tile {}, bufferize {:?}: synthesis failed",
                    shape_str(&tile_output_shape),
                    bufferize,
                );
                continue;
            };

            let cost = Cost::from_impl(&implementation).main;
            let Ok(result) = implementation.bench(ITERS, None) else {
                eprintln!(
                    "skipping tile {}, bufferize {:?}: benchmark failed",
                    shape_str(&tile_output_shape),
                    bufferize,
                );
                continue;
            };
            let runtime_seconds =
                (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();

            writer.write_record([
                implementation_id.to_string(),
                shape_str(&tile_output_shape),
                bufferize.index.to_string(),
                bufferize.memory.to_string(),
                bufferize.layout.to_string(),
                bufferize
                    .vector_size
                    .map(DimSize::get)
                    .unwrap_or(0)
                    .to_string(),
                cost.to_string(),
                runtime_seconds.to_string(),
            ])?;
            implementation_id += 1;
        }
    }

    writer.flush()?;
    Ok(())
}

fn apply_bufferize_to_default_leaf(
    node: ImplNode<Avx2Target>,
    bufferize: &Bufferize<Avx2Target>,
) -> Result<ImplNode<Avx2Target>, ApplyError> {
    match node {
        ImplNode::SpecApp(app) => Ok(ImplNode::FunctionApp(FunctionApp {
            body: Box::new(bufferize.apply(&app.0)?),
            parameters: app.1.clone(),
            spec: Some(app.0.clone()),
        })),
        _ => {
            let children = node.children();
            match children {
                [] => Ok(node),
                [_] => {
                    let child = apply_bufferize_to_default_leaf(children[0].clone(), bufferize)?;
                    Ok(node.replace_children(std::iter::once(child)))
                }
                _ => {
                    let default_child_idx = node.default_child().ok_or(
                        ApplyError::NotApplicable(morello::scheduling::NotApplicableReason::Other(
                            Some("No default child"),
                        )),
                    )?;
                    let new_children = children
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(idx, child)| {
                            if idx == default_child_idx {
                                apply_bufferize_to_default_leaf(child, bufferize)
                            } else {
                                Ok(child)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(node.replace_children(new_children.into_iter()))
                }
            }
        }
    }
}

fn synthesize_all_impls(
    node: ImplNode<Avx2Target>,
    db: &morello::db::FilesDatabase,
) -> Option<ImplNode<Avx2Target>> {
    match node {
        ImplNode::SpecApp(app) => {
            eprintln!("synthesizing: {}", app.0);
            top_down_many_impls(db, std::slice::from_ref(&app.0))
                .into_iter()
                .exactly_one()
                .unwrap()
                .into_iter()
                .next()
        }
        _ => {
            let mut synthed = node
                .children()
                .iter()
                .map(|c| synthesize_all_impls(c.clone(), db))
                .collect::<Option<Vec<_>>>()?
                .into_iter();
            Some(
                node.map_children(|children| children.into_iter().map(|_| synthed.next().unwrap())),
            )
        }
    }
}

fn make_spec() -> Spec<Avx2Target> {
    let inner_matmul = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![BATCH, N, K, KO],
        dtypes: vec![DTYPE; 3],
    };
    let outer_matmulaccum = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: true },
        spec_shape: shape![BATCH, N, KO, N],
        dtypes: vec![DTYPE; 3],
    };

    let operand_auxes = [
        &shape![BATCH, KO, N],
        &shape![BATCH, N, K],
        &shape![BATCH, K, KO],
        &shape![BATCH, N, N],
    ]
    .into_iter()
    .map(|shape| TensorSpecAux {
        memory: GL,
        layout: row_major(shape),
        vector_size: None,
    })
    .collect();

    let mut spec = Spec::<Avx2Target>(
        LogicalSpec::Compose {
            components: vec![outer_matmulaccum, inner_matmul],
            operand_auxes,
            serial_only: true,
        },
        Avx2Target::max_mem(),
    );
    spec.canonicalize().unwrap();
    spec
}

/// Returns options for `bufferize`'ing the given `Compose` [Spec].
///
/// If `spec` is not a `Compose`, then this returns an empty [Vec].
fn bufferize_params(spec: &Spec<Avx2Target>) -> Vec<Bufferize<Avx2Target>> {
    let LogicalSpec::Compose { components, .. } = &spec.0 else {
        unreachable!("expected a Compose spec");
    };
    assert_eq!(components.len(), 2);

    let mut result = vec![];
    let intermediate_shape = components[0].parameter_shape(0);
    for memory in Avx2Target::memories() {
        for layout in Avx2Target::move_destination_layouts(&intermediate_shape, DTYPE) {
            if !memory.vector_bytes().is_empty() {
                for vector_size in vector_sizes(DTYPE, memory.vector_bytes()) {
                    result.push(Bufferize {
                        index: 0,
                        memory,
                        layout: layout.clone(),
                        vector_size: Some(vector_size),
                    });
                }
            } else {
                result.push(Bufferize {
                    index: 0,
                    memory,
                    layout: layout.clone(),
                    vector_size: None,
                });
            }
        }
    }
    result
}

fn vector_sizes(dtype: Dtype, vector_bytes: &'static [u32]) -> impl Iterator<Item = DimSize> {
    vector_bytes.iter().map(move |&bytes| {
        DimSize::new(bytes / u32::from(dtype.size())).expect("vector size should not be zero")
    })
}

fn shape_str(shape: &[DimSize]) -> String {
    shape.iter().map(|d| d.get().to_string()).join("x")
}

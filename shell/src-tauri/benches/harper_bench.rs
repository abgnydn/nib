//! Harper lint throughput: `wire::check_text_with` over a shared linter
//! built once via `state::build_linter`.
//!
//! Run:
//!     cargo bench --features llm
//! Compile-only (CI-fast):
//!     cargo bench --no-run --features llm

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nib_lib::{state::build_linter, wire::check_text_with};

/// Repeat `base` and truncate to exactly `n` chars (ASCII, so chars == bytes).
fn make_input(base: &str, n: usize) -> String {
    base.chars()
        .cycle()
        .take(n)
        .collect()
}

fn bench_check_text(c: &mut Criterion) {
    const BASE: &str = "This is an test of the Harper grammer checker with a few extra words for length. ";
    let inputs: Vec<(&str, String)> = vec![
        ("short", make_input(BASE, 50)),
        ("medium", make_input(BASE, 500)),
        ("long", make_input(BASE, 2000)),
    ];
    for (label, len) in [("short", 50), ("medium", 500), ("long", 2000)] {
        let found = inputs.iter().find(|(l, _)| *l == label).unwrap();
        debug_assert_eq!(found.1.chars().count(), len, "{label} input must be {len} chars");
    }

    let mut group = c.benchmark_group("harper_check");
    for (id, text) in &inputs {
        group.bench_with_input(BenchmarkId::from_parameter(id), text, |b, text| {
            let mut linter = build_linter();
            b.iter(|| {
                let lints = check_text_with(&mut linter, black_box(text));
                black_box(lints);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_check_text);
criterion_main!(benches);

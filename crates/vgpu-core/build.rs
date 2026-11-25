//! Embeds the git commit and whether the code compiled into this binary had
//! uncommitted changes, so every prediction can say which code produced it.
//! Python code is not compiled in; callers check its state themselves.

use std::process::Command;

/// Repository paths compiled into the binary.
const COMPILED: [&str; 4] = ["crates", "schemas", "Cargo.toml", "Cargo.lock"];

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let root = git(&["rev-parse", "--show-toplevel"]);
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let mut status = vec![
        "-C",
        root.as_deref().unwrap_or("."),
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
    ];
    status.extend(COMPILED);
    let dirty = git(&status).is_none_or(|s| !s.is_empty());
    println!("cargo:rustc-env=VGPU_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=VGPU_BUILD_DIRTY={dirty}");

    // Rerun when the commit or the index moves, or a compiled file changes.
    if let (Some(git_dir), Some(common)) = (
        git(&["rev-parse", "--absolute-git-dir"]),
        git(&["rev-parse", "--git-common-dir"]),
    ) {
        for p in [
            format!("{git_dir}/HEAD"),
            format!("{git_dir}/index"),
            // Appended on every commit, checkout and reset in this worktree,
            // so a new commit is seen even when no watched file's mtime moves.
            format!("{git_dir}/logs/HEAD"),
            format!("{common}/refs/heads"),
            // Branch tips after `git gc` or `git pack-refs`.
            format!("{common}/packed-refs"),
        ] {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    if let Some(root) = root {
        for p in COMPILED {
            println!("cargo:rerun-if-changed={root}/{p}");
        }
    }
}

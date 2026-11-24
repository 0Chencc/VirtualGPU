//! Pairwise verdicts (`docs/design.md`): A is faster than B only if A is at
//! least 5% faster in every draw. Anything else is indistinguishable, because
//! profile perturbation does not cover the model's structural error and 5%
//! is the practical band used on hardware.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    AFaster,
    BFaster,
    Indistinguishable,
}

/// Fixed model resolution: `|ln(t_a / t_b)| >= ln(1.05)` in every draw.
pub const RESOLUTION: f64 = 1.05;

/// Verdict from the times of A and B under the same draws, in draw order.
pub fn verdict(a: &[f64], b: &[f64]) -> Verdict {
    assert_eq!(a.len(), b.len(), "both variants need the same draws");
    let band = RESOLUTION.ln();
    let ratios = a.iter().zip(b).map(|(x, y)| (x / y).ln());
    let (mut a_faster, mut b_faster) = (true, true);
    for r in ratios {
        a_faster &= r <= -band;
        b_faster &= r >= band;
    }
    match (a_faster && !a.is_empty(), b_faster && !a.is_empty()) {
        (true, _) => Verdict::AFaster,
        (_, true) => Verdict::BFaster,
        _ => Verdict::Indistinguishable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_five_percent_in_every_draw() {
        assert_eq!(verdict(&[90.0, 94.0], &[100.0, 100.0]), Verdict::AFaster);
        assert_eq!(verdict(&[100.0, 100.0], &[90.0, 94.0]), Verdict::BFaster);
        // One draw inside the band.
        assert_eq!(
            verdict(&[90.0, 96.0], &[100.0, 100.0]),
            Verdict::Indistinguishable
        );
        // Directions disagree.
        assert_eq!(
            verdict(&[90.0, 110.0], &[100.0, 100.0]),
            Verdict::Indistinguishable
        );
        assert_eq!(verdict(&[], &[]), Verdict::Indistinguishable);
    }
}

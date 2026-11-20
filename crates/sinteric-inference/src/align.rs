/// `n` rounded up to a multiple of `align`, a power of two.
#[inline]
pub fn align_up(n: usize, align: usize) -> crate::Result<usize> {
    if !align.is_power_of_two() {
        return Err(crate::Error::config(format!("invalid alignment {align}")));
    }
    n.checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .ok_or(crate::Error::size_overflow("alignment"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_the_boundary() {
        assert_eq!(align_up(0, 64).unwrap(), 0);
        assert_eq!(align_up(1, 64).unwrap(), 64);
        assert_eq!(align_up(64, 64).unwrap(), 64);
        assert_eq!(align_up(65, 64).unwrap(), 128);
    }
}

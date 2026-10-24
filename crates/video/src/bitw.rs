//! Bit writer for the parameter sets written here: RBSP bits in, a NAL unit with emulation prevention bytes out.

#[derive(Default)]
pub struct W {
    b: Vec<u8>, // whole bytes
    c: u64,     // bits not yet in `b`: the low `n`
    n: u32,
}

impl W {
    /// `v` in `n` ≤ 32 bits.
    pub fn u(&mut self, n: u32, v: u32) -> &mut Self {
        self.c = self.c << n | (v as u64 & ((1 << n) - 1));
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.b.push((self.c >> self.n) as u8);
        }
        self
    }

    pub fn f(&mut self, v: bool) -> &mut Self {
        self.u(1, v as u32)
    }

    /// Exp-Golomb, `v` < 2^31.
    pub fn ue(&mut self, v: u32) -> &mut Self {
        let z = 31 - (v + 1).leading_zeros();
        self.u(z, 0).u(z + 1, v + 1)
    }

    /// rbsp_trailing_bits, then the bytes with an emulation prevention byte after each 00 00 that a byte ≤ 3 follows.
    pub fn nal(&mut self) -> Vec<u8> {
        self.u(1, 1);
        if self.n > 0 {
            self.u(8 - self.n, 0);
        }
        let (mut o, mut z) = (Vec::with_capacity(self.b.len() + 4), 0);
        for &x in &self.b {
            if z >= 2 && x <= 3 {
                o.push(3);
                z = 0;
            }
            z = if x == 0 { z + 1 } else { 0 };
            o.push(x);
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::Bits;

    #[test]
    fn round_trip() {
        let mut w = W::default();
        w.u(8, 0x67)
            .ue(0)
            .ue(1)
            .ue(254)
            .f(true)
            .u(32, 0)
            .u(32, 0)
            .ue(7)
            .u(3, 5);
        let b = w.nal();
        // 32 zero bits in a row need emulation prevention
        assert!(b.windows(3).all(|x| x[..2] != [0, 0] || x[2] >= 3) && b.contains(&3));
        let mut r = Bits::new(&b);
        let v = [
            r.u(8),
            r.ue(),
            r.ue(),
            r.ue(),
            r.u(1),
            r.u(32),
            r.u(32),
            r.ue(),
            r.u(3),
        ];
        assert_eq!(v, [0x67, 0, 1, 254, 1, 0, 0, 7, 5]);
        assert!(!r.more() && !r.end());
    }
}

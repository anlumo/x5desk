//! Bit reader over a NAL unit payload that drops emulation prevention bytes as it goes (no RBSP copy).
//! Reads past the end return zeros; parsers check `end()` once at the end.

pub struct Bits<'a> {
    b: &'a [u8],
    i: usize,     // next raw byte
    z: u8,        // zero bytes right before i
    c: u8,        // current byte
    k: u8,        // bits left in c
    pub p: usize, // RBSP bits read
}

impl<'a> Bits<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Bits {
            b,
            i: 0,
            z: 0,
            c: 0,
            k: 0,
            p: 0,
        }
    }

    fn bit(&mut self) -> u32 {
        if self.k == 0 {
            let mut x = self.b.get(self.i).copied().unwrap_or(0);
            if self.z >= 2 && x == 3 {
                self.i += 1;
                self.z = 0;
                x = self.b.get(self.i).copied().unwrap_or(0);
            }
            self.z = if x == 0 { self.z.saturating_add(1) } else { 0 };
            (self.i, self.c, self.k) = (self.i + 1, x, 8);
        }
        self.k -= 1;
        self.p += 1;
        (self.c >> self.k & 1) as u32
    }

    pub fn u(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| v << 1 | self.bit())
    }

    pub fn f(&mut self) -> bool {
        self.bit() != 0
    }

    /// u32::MAX for a code longer than 32 bits (only garbage or reading past the end makes one).
    pub fn ue(&mut self) -> u32 {
        let mut z = 0;
        while self.bit() == 0 {
            z += 1;
            if z == 32 {
                return u32::MAX;
            }
        }
        ((1u64 << z) - 1 + self.u(z) as u64) as u32
    }

    pub fn se(&mut self) -> i32 {
        let k = self.ue() as i64;
        (if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) }) as i32
    }

    pub fn skip(&mut self, n: u32) {
        for _ in 0..n {
            self.bit();
        }
    }

    pub fn end(&self) -> bool {
        self.i > self.b.len()
    }

    /// more_rbsp_data(): is anything left before the stop bit?
    pub fn more(&self) -> bool {
        let Some(l) = self.b.iter().rposition(|&x| x != 0) else {
            return false;
        };
        let stop = l * 8 + 7 - self.b[l].trailing_zeros() as usize;
        let epb = self.k == 0 && self.z >= 2 && self.b.get(self.i) == Some(&3);
        (self.i + epb as usize) * 8 - (self.k as usize) < stop
    }
}

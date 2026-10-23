//! CPython's `random.Random`, reimplemented.
//!
//! This is a port rather than "a Mersenne Twister", and the difference matters. A game
//! in this project is a pure function of `(ruleset_hash, seed, agent_configs)`, and
//! every recorded game, every calibration number and every published index in
//! CLAUDE.md was produced by agents drawing from `random.Random`. Swapping in a
//! different generator would keep determinism and silently move every measured
//! baseline, which is a bad trade for a few nanoseconds.
//!
//! Reproduced exactly: `init_by_array` seeding from an integer, `getrandbits`,
//! `_randbelow_with_getrandbits`, `choice` and `shuffle`. Together those are every
//! draw an agent makes.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct PyRandom {
    state: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Seed the way `random.Random(n)` does for a non-negative integer: the absolute
    /// value is split into 32-bit little-endian words and fed to `init_by_array`.
    pub fn new(seed: u64) -> Self {
        let mut key: Vec<u32> = Vec::with_capacity(2);
        if seed == 0 {
            key.push(0);
        } else {
            let mut remaining = seed;
            while remaining != 0 {
                key.push((remaining & 0xFFFF_FFFF) as u32);
                remaining >>= 32;
            }
        }
        let mut rng = Self {
            state: [0; N],
            index: N,
        };
        rng.init_by_array(&key);
        rng
    }

    /// The per-seat stream `tournament.play.seat_rng` derives, so that how much
    /// randomness one agent consumes cannot shift what the other draws.
    pub fn for_seat(seed: u64, seat: usize) -> Self {
        Self::new((seed << 1) ^ (seat as u64 + 0x9E37_79B9))
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let previous = self.state[i - 1];
            self.state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key.len());
        while k > 0 {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.state[0] = 0x8000_0000;
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            for i in 0..N {
                let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
                let mut next = self.state[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.state[i] = next;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.getrandbits(k)` for `k <= 64`, which covers every draw made here: the
    /// only caller is `randbelow`, and its argument is a move-list length.
    pub fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        if bits <= 32 {
            return (self.genrand_u32() >> (32 - bits)) as u64;
        }
        let low = self.genrand_u32() as u64;
        let high = (self.genrand_u32() >> (64 - bits)) as u64;
        low | (high << 32)
    }

    /// `random._randbelow_with_getrandbits`: rejection sampling on the bit length, so
    /// the number of words consumed matches CPython draw for draw.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let bits = 64 - n.leading_zeros();
        loop {
            let value = self.getrandbits(bits);
            if value < n {
                return value;
            }
        }
    }

    pub fn choice<T: Copy>(&mut self, sequence: &[T]) -> T {
        sequence[self.randbelow(sequence.len() as u64) as usize]
    }

    pub fn shuffle<T>(&mut self, sequence: &mut [T]) {
        for i in (1..sequence.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            sequence.swap(i, j);
        }
    }

    /// `random.random()`: 53 bits of precision, as CPython builds it.
    ///
    /// No agent draws a float -- every choice goes through `randbelow` -- but the unit
    /// test below pins it, because it is the shortest check that the twist and the
    /// tempering are right independently of the rejection sampling on top.
    #[allow(dead_code)]
    pub fn random(&mut self) -> f64 {
        let a = self.genrand_u32() >> 5;
        let b = self.genrand_u32() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    /// Every expectation here was produced by CPython's own `random` module. They are
    /// the artefact, exactly as the perft node counts are: if this file drifts, the
    /// agents stop reproducing recorded games and every calibration number moves.
    #[test]
    fn matches_cpython_random() {
        let mut rng = PyRandom::new(0);
        let expected = [0.8444218515250481, 0.7579544029403025, 0.420571580830845];
        for value in expected {
            assert_eq!(rng.random(), value);
        }
    }

    #[test]
    fn matches_cpython_getrandbits() {
        let mut rng = PyRandom::new(12345);
        let widths = [1u32, 7, 32, 33, 64, 53];
        let expected = [0u64, 93, 43_676_229, 7_817_590_892, 5_508_932_873_585_391_847, 3_318_359_830_190_266];
        for (bits, want) in widths.into_iter().zip(expected) {
            assert_eq!(rng.getrandbits(bits), want);
        }
    }

    #[test]
    fn matches_cpython_randbelow() {
        let mut rng = PyRandom::new(7);
        let bounds = [3u64, 10, 100, 7, 1_000_000];
        let expected = [1u64, 2, 50, 5, 50_631];
        for (bound, want) in bounds.into_iter().zip(expected) {
            assert_eq!(rng.randbelow(bound), want);
        }
    }

    #[test]
    fn matches_cpython_shuffle() {
        let mut rng = PyRandom::new(99);
        let mut values: Vec<u32> = (0..10).collect();
        rng.shuffle(&mut values);
        assert_eq!(values, vec![7, 2, 0, 8, 5, 1, 4, 3, 9, 6]);
    }

    #[test]
    fn matches_cpython_for_a_seat_stream() {
        let mut rng = PyRandom::for_seat(5, 1);
        let drawn: Vec<u64> = (0..5).map(|_| rng.randbelow(20)).collect();
        assert_eq!(drawn, vec![9, 15, 0, 9, 9]);
    }

    /// A seed wider than 32 bits takes two key words through `init_by_array`, which is
    /// the branch a tournament reaches once its seeds get large.
    #[test]
    fn matches_cpython_for_a_wide_seed() {
        let mut rng = PyRandom::new((1u64 << 40) + 12345);
        let drawn: Vec<u64> = (0..4).map(|_| rng.randbelow(1000)).collect();
        assert_eq!(drawn, vec![317, 357, 117, 3]);
    }

    /// F10b (Q752b): `getrandbits(0)` is CPython's 0 and draws no word; `randbelow(0)`,
    /// which no caller asks (a `choice` from nothing is never made) and on which CPython
    /// would loop, answers 0 and draws none either. The stream after both is the fresh one.
    #[test]
    fn zero_bits_and_a_zero_bound_draw_nothing() {
        let mut drawn = PyRandom::new(3);
        let mut fresh = PyRandom::new(3);
        assert_eq!(drawn.getrandbits(0), 0);
        assert_eq!(drawn.randbelow(0), 0);
        assert_eq!(drawn.random(), fresh.random());
    }
}

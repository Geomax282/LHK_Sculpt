//! Minimal interleaved image container (row-major, HxWxC), like numpy/OpenCV
//! arrays.

#[derive(Clone, Debug, PartialEq)]
pub struct Mat<T> {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Mat<T> {
    pub fn new(h: usize, w: usize, c: usize) -> Self {
        Self { h, w, c, data: vec![T::default(); h * w * c] }
    }

    pub fn filled(h: usize, w: usize, c: usize, v: T) -> Self {
        Self { h, w, c, data: vec![v; h * w * c] }
    }

    pub fn from_vec(h: usize, w: usize, c: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), h * w * c, "Mat::from_vec size mismatch");
        Self { h, w, c, data }
    }

    #[inline]
    pub fn at(&self, y: usize, x: usize, ch: usize) -> T {
        self.data[(y * self.w + x) * self.c + ch]
    }

    #[inline]
    pub fn set(&mut self, y: usize, x: usize, ch: usize, v: T) {
        let i = (y * self.w + x) * self.c + ch;
        self.data[i] = v;
    }

    pub fn row(&self, y: usize) -> &[T] {
        let s = y * self.w * self.c;
        &self.data[s..s + self.w * self.c]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Extract one channel as a single-channel Mat.
    pub fn channel(&self, ch: usize) -> Mat<T> {
        let data = (0..self.h * self.w).map(|i| self.data[i * self.c + ch]).collect();
        Mat::from_vec(self.h, self.w, 1, data)
    }

    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Mat<U> {
        Mat::from_vec(self.h, self.w, self.c, self.data.iter().map(|&v| f(v)).collect())
    }
}

impl Mat<u8> {
    pub fn to_f32(&self) -> Mat<f32> {
        self.map(|v| v as f32)
    }
}

impl Mat<u16> {
    pub fn to_f32(&self) -> Mat<f32> {
        self.map(|v| v as f32)
    }
}

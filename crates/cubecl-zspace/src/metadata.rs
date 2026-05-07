use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    MetadataError, shape::Shape, strides::Strides, striding::row_major_contiguous_strides,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Metadata {
    pub shape: Shape,
    pub strides: Strides,
    pub tiler: Option<Tiler>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Tiler {
    start_axis: u8, // 0..256, représente un entier < rank
    tile_size: SmallVec<[u16; 3]>, // 0..65k
                    // 16 * 3 + 8 = 56 + Option: ~8 -> 64 bits
}

impl Metadata {
    pub fn new(shape: impl Into<Shape>, strides: impl Into<Strides>) -> Self {
        let shape = shape.into();
        let strides = strides.into();
        debug_assert_eq!(
            shape.rank(),
            strides.rank(),
            "Rank of shape and strides must be the same"
        );

        Self {
            shape,
            strides,
            tiler: None,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn shape_mut(&mut self) -> &mut Shape {
        &mut self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn strides_mut(&mut self) -> &mut Strides {
        &mut self.strides
    }

    pub fn rank(&self) -> usize {
        self.num_dims()
    }

    pub fn num_dims(&self) -> usize {
        self.shape.num_dims()
    }

    /// Returns the total number of elements of a tensor having this shape
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn swapped(mut self, dim0: usize, dim1: usize) -> Self {
        self.swap(dim0, dim1);
        self
    }

    pub fn swap(&mut self, dim0: usize, dim1: usize) {
        debug_assert!(dim0 < self.rank(), "dim0 is out of bounds");
        debug_assert!(dim1 < self.rank(), "dim1 is out of bounds");
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
        self.shape.permute(axes)?;
        self.strides.permute(axes)?;

        Ok(())
    }

    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, MetadataError> {
        self.permute(axes)?;
        Ok(self)
    }

    /// Insert a dimension of `shape` with `stride` at position `index`.
    pub fn insert(&mut self, index: usize, shape: usize, stride: usize) {
        self.shape.insert(index, shape);
        self.strides.insert(index, stride);
    }

    /// Remove and return the dimension at position `index` from the metadata.
    pub fn remove(&mut self, index: usize) -> (usize, usize) {
        let shape = self.shape.remove(index);
        let stride = self.strides.remove(index);
        (shape, stride)
    }

    /// Appends a dimension of `shape` with `stride` to the back of the metadata.
    pub fn push(&mut self, shape: usize, stride: usize) {
        self.shape.push(shape);
        self.strides.push(stride);
    }

    pub fn to_tiled(&self, start_axis: u8, tile: &[u16]) -> Self {
        let start_axis = start_axis as usize;
        let mut new_metadata = Metadata::new(Shape::new([]), Strides::new(&[]));
        for i in 0..start_axis {
            let dim = self.shape[i];
            new_metadata.push(dim, 0);
        }

        let mut i = 0;
        #[allow(clippy::explicit_counter_loop)]
        for j in start_axis..tile.len() {
            assert!(self.shape[j].is_multiple_of(tile[i] as usize), "self.shape[{j}] must be divisible by tile[{i}]");
            let dim = self.shape[j] / tile[i] as usize;
            new_metadata.push(dim, 0);
            i += 1;
        }
        i = 0;
        for _ in start_axis + tile.len()..start_axis + 2 * tile.len() {
            let dim = tile[i] as usize;
            new_metadata.push(dim, 0);
            i += 1;
        }
        for i in start_axis + tile.len()..self.shape.len() {
            let dim = self.shape[i];
            new_metadata.push(dim, 0);
        }
        new_metadata.strides = row_major_contiguous_strides(&new_metadata.shape);
        new_metadata.tiler = Some(Tiler {
            start_axis: start_axis as u8,
            tile_size: SmallVec::from_slice(tile),
        });

        new_metadata
    }
}

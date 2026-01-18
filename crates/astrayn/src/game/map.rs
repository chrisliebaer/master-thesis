use std::collections::HashMap;

use glam::IVec2;

/// The size of a chunk in tiles across one dimension.
const CHUNK_SIZE: usize = 128;

/// Type alias for vertical cell height.
type CellHeight = i16;

/// Type alias for the 2d grid of cells that form a chunk.
type ChunkGrid = [[Cell; CHUNK_SIZE]; CHUNK_SIZE];

/// Type alias for the index of a chunk in a map.
type ChunkIndex = IVec2;

/// Type alias for the index of a cell in a chunk.
type CellIndex = (u8, u8);

/// A map forms an isolated part of the game world.
///
/// As a general rule, there are no interactions between maps outside scripted events.
/// Maps possess their own physic space.
pub struct Map {
	/// The chunks that form the map.
	chunks: HashMap<ChunkIndex, Chunk>,
}

/// A chunk forms a part of a map.
///
/// It is a square grid of cells, each with a height.
/// Chunks form atomic units of the map, and are loaded and unloaded as a whole.
#[derive(Clone, Debug)]
pub struct Chunk {
	grid: ChunkGrid,
}

/// A cell forms a part of a chunk.
///
/// It stores a lot of static information about the state of the cell.
#[derive(Clone, Debug)]
pub struct Cell {
	/// The height of the cell.
	height: CellHeight,
}

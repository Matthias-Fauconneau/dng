use {image::{xy, Image}, bytemuck::{Zeroable, Pod, cast, cast_slice}};

pub fn gain(code: &[u8]) -> [Image<Box<[f32]>>; 4] {
	let code = cast_slice::<_,u32>(code);
	let [len, mut ref code @ ..] = code[..] else {panic!()};
	let len = u32::from_be(len);
	*(0..len).map(|_| {
		#[repr(transparent)] #[allow(non_camel_case_types)] #[derive(Clone,Copy,Zeroable,Pod)] struct u32be(u32);
		impl From<u32be> for u32 { fn from(u32be(be): u32be) -> Self { Self::from_be(be) } }
		impl std::fmt::Debug for u32be { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { u32::from(*self).fmt(f) } }
		#[repr(transparent)] #[allow(non_camel_case_types)] #[derive(Clone,Copy,Zeroable,Pod)] struct f32be([u8; 4]);
		impl From<f32be> for f32 { fn from(f32be(be): f32be) -> Self { Self::from_be_bytes(be) } }
		impl std::fmt::Debug for f32be { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f32::from(*self).fmt(f) } }
		#[repr(transparent)] #[allow(non_camel_case_types)] #[derive(Clone,Copy,Zeroable,Pod)] struct f64be([u8; 8]);
		impl From<f64be> for f64 { fn from(f64be(be): f64be) -> Self { Self::from_be_bytes(be) } }
		impl std::fmt::Debug for f64be { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f64::from(*self).fmt(f) } }
		#[repr(C,packed)] #[derive(Debug,Clone,Copy,Zeroable,Pod)] struct GainMap {
			id: u32be,
			version: u32be,
			flags: u32be,
			len: u32be,
			top: u32be,
			left: u32be,
			bottom: u32be,
			right: u32be,
			plane: u32be,
			planes: u32be,
			row_pitch: u32be,
			column_pitch: u32be,
			size_y: u32be,
			size_x: u32be,
			map_spacing_vertical: f64be,
			map_spacing_horizontal: f64be,
			map_origin_vertical: f64be,
			map_origin_horizontal: f64be,
			map_planes: u32be,
		}
		let opcode : &[_; 23]; (opcode, code) = code.split_first_chunk().unwrap();
		let GainMap{id, map_planes, size_x, size_y, ..} = cast::<_,GainMap>(*opcode);
		assert_eq!(u32::from(id), 9);
		assert_eq!(u32::from(map_planes), 1);
		let size = xy{x: size_x.into(), y: size_y.into()};
		let gain; (gain, code) = code.split_at((size.y*size.x) as usize);
		Image::new(size, cast_slice::<_, f32be>(gain)).map(|&gain| f32::from(gain))
	}).collect::<Box<_>>().into_array().unwrap()
}
use core::simd::{cmp::SimdPartialOrd, num::SimdUint, u8x64, u32x64, mask32x64};
const ZERO : u32x64  = u32x64::from_array([0; _]);
//const INDEX: u8x64 = u8x64::from_array(std::array::from_fn(|i| i as u8));
const INDEX: u8x64 = u8x64::from_array([
0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x3B,0x3C,0x3D,0x3E,0x3F]);
use {vector::{xy, minmax, MinMax}, image::{Image, XYZ}};
pub fn adaptive_histogram_equalization(image: &Image<impl AsRef<[XYZ<f32>]>>, radius: u32) -> Image<Box<[u32]>> { // TODO: SIMD
	let radius = radius as i32;
	let MinMax{min, max} = minmax(image.data.as_ref().iter().map(|&XYZ{Y,..}| Y)).unwrap();
	const coarse : usize = core::simd::u32x64::LEN;
	const fine : usize = core::simd::u32x64::LEN;
	const Nbins : usize = coarse*fine;
	//assert!(image.size.x <= image.size.y); // FIXME: benchmark compute vs bandwidth tradeoff
	assert!(image.size.x as usize*(coarse*fine)*4 <= 32*1024*1024);
	let luminance = image.as_ref().map(|XYZ{Y,..}| (f32::ceil((Y-min)/(max-min)*(Nbins as f32)-1.)) as u16);
	struct Histogram { sums: u32x64, bins: [u32x64; coarse] }
	let mut columns = unsafe{Box::<[Histogram]>::new_zeroed_slice(luminance.size.x as usize).assume_init()};
	let mut rank = Image::zero(luminance.size);
	let ([w, h], stride) = (luminance.size.signed().into(), luminance.stride as i32);
	assert_eq!(rank.stride as i32, stride);
	for y in -radius..=radius { for x in 0..luminance.size.x {
		let bin = luminance[xy{x,y: y.max(0) as u32}] as usize;
		let ref mut column = columns[x as usize];
		column.sums[bin/fine] += 1;
		column.bins[bin/fine][bin%fine] += 1;
	} }
	let start = std::time::Instant::now();
	let mut y = 0;
	loop {
		if !(y < h) { break; }
		println!("{y}");
		let Histogram{mut sums, mut bins} = Histogram{sums: ZERO, bins: [ZERO; _]};
		for x in -radius..=radius { 
			let ref column = columns[x.max(0) as usize];
			sums += column.sums;
			for segment in 0..coarse { bins[segment] += column.bins[segment]; }
		}
		for x in 0..w-1 {
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = // Lookup masks?
				mask32x64::from(INDEX.simd_lt(u8x64::splat((bin/fine) as u8))).select(sums, ZERO).reduce_sum() + 
				mask32x64::from(INDEX.simd_le(u8x64::splat((bin%fine) as u8))).select(bins[bin/fine], ZERO).reduce_sum();
			// Slide right
			let ref right = columns[(x+radius+1).min(w-1) as usize];
			let ref left = columns[(x-radius).max(0) as usize];
			sums += right.sums - left.sums;
			for segment in 0..coarse {
				if right.sums[segment] > 0 { bins[segment] += right.bins[segment]; }
				if left.sums[segment] > 0 { bins[segment] -= left.bins[segment]; }
			}
		}
		{ // Last of row iteration (not sliding further right after)
			let x = w-1;
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
		}
		// Slide down
		for x in 0..w {
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			column.bins[bin/fine][bin%fine] -= 1;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			column.bins[bin/fine][bin%fine] += 1;
		}
		y += 1;
		if !(y < h) { break; }
		let Histogram{mut sums, mut bins} = Histogram{sums: [0; _].into(), bins: [[0; _].into(); _]};
		for x in (w-1)-radius..=(w-1)+radius {
			let ref column = columns[x.min(w-1) as usize];
			sums += column.sums;
			for segment in 0..coarse { bins[segment] += column.bins[segment]; }
		}
		for x in (1..w).into_iter().rev() {
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = // Lookup masks?
				mask32x64::from(INDEX.simd_lt(u8x64::splat((bin/fine) as u8))).select(sums, ZERO).reduce_sum() + 
				mask32x64::from(INDEX.simd_le(u8x64::splat((bin%fine) as u8))).select(bins[bin/fine], ZERO).reduce_sum();
			// Slide left
			let ref left = columns[(x-radius-1).max(0) as usize];
			let ref right = columns[(x+radius).min(w-1) as usize];
			sums += left.sums - right.sums;
			for segment in 0..coarse {
				if left.sums[segment] > 0 { bins[segment] += left.bins[segment]; }
				if right.sums[segment] > 0 { bins[segment] -= right.bins[segment]; }
			}
		}
		{ // Back to first of row iteration (not sliding further left after)
			let x = 0;
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
		}
		// Slide down
		for x in 0..w {
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			column.bins[bin/fine][bin%fine] -= 1;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			column.bins[bin/fine][bin%fine] += 1;
		}
		y += 1;
	}
	println!("{}ms", start.elapsed().as_millis());
	rank
}

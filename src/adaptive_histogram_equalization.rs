use core::simd::{cmp::SimdPartialOrd, num::SimdUint, u8x64, u32x64, mask32x64};
const ZERO : u32x64  = u32x64::from_array([0; _]);
const fn enumerate<const N: usize>() -> [u8; N] { let mut a = [0; N]; let mut i = 0; while i < N { a[i] = i as _; i += 1; } a }
const INDEX: u8x64 = u8x64::from_array(enumerate());
pub struct Instant(u64);
pub fn now() -> Instant { Instant(unsafe{core::arch::x86_64::_rdtsc()}) }
#[derive(Clone,Copy)] pub struct Duration(u64);
pub fn fence(since: Instant, elapsed: &mut Duration) -> Instant { let now = now(); elapsed.0 += now.0-since.0; now }
macro_rules! id { [$($id:ident),+] => { [ $((std::stringify!($id), *$id)),+ ] } }
use {vector::{xy, minmax, MinMax}, image::{Image, XYZ}};
pub fn adaptive_histogram_equalization(image: &Image<impl AsRef<[XYZ<f32>]>>, radius: u32) -> Image<Box<[u32]>> { // TODO: SIMD
	let start_time = std::time::Instant::now();
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
	let Histogram{mut sums, mut bins} = Histogram{sums: ZERO, bins: [ZERO; _]};
	for x in -radius..=radius {
		let ref column = columns[x.max(0) as usize];
		sums += column.sums;
		for segment in 0..coarse { bins[segment] += column.bins[segment]; }
	}
	let start = now();
	let [ref mut down, ref mut forward, ref mut reverse] = [Duration(0); _]; 
	let mut last_start = now();
	let mut y = 0;
	loop {
		if !(y < h) { break; }
		let start = fence(last_start, down);
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
		let start = fence(start, forward);
		{ // Last of row iteration (not sliding further right after)
			let x = w-1;
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
		}
		// Slide down
		for x in 0..(w-1)-radius {
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			column.bins[bin/fine][bin%fine] -= 1;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			column.bins[bin/fine][bin%fine] += 1;
		}
		for x in (w-1)-radius..w-1 {
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			sums[bin/fine] -= 1;
			column.bins[bin/fine][bin%fine] -= 1;
			bins[bin/fine][bin%fine] -= 1;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			sums[bin/fine] += 1;
			column.bins[bin/fine][bin%fine] += 1;
			bins[bin/fine][bin%fine] += 1;
		}
		{
			let x = w - 1;
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			sums[bin/fine] -= 1+radius as u32;
			column.bins[bin/fine][bin%fine] -= 1;
			bins[bin/fine][bin%fine] -= 1+radius as u32;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			sums[bin/fine] += 1+radius as u32;
			column.bins[bin/fine][bin%fine] += 1;
			bins[bin/fine][bin%fine] += 1+radius as u32;
		}
		y += 1;
		if !(y < h) { break; }
		let start = fence(start, down);
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
		last_start = fence(start, reverse);
		{ // Back to first of row iteration (not sliding further left after)
			let x = 0;
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
		}
		// Slide down
		{
			let x = 0;
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			sums[bin/fine] -= 1+radius as u32;
			column.bins[bin/fine][bin%fine] -= 1;
			bins[bin/fine][bin%fine] -= 1+radius as u32;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			sums[bin/fine] += 1+radius as u32;
			column.bins[bin/fine][bin%fine] += 1;
			bins[bin/fine][bin%fine] += 1+radius as u32;
		}
		for x in 1..=radius {
			let bin = luminance[((y-radius).max(0)*stride+x) as usize] as usize;
			let ref mut column = columns[x as usize];
			column.sums[bin/fine] -= 1;
			sums[bin/fine] -= 1;
			column.bins[bin/fine][bin%fine] -= 1;
			bins[bin/fine][bin%fine] -= 1;
			let bin = luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize;
			column.sums[bin/fine] += 1;
			sums[bin/fine] += 1;
			column.bins[bin/fine][bin%fine] += 1;
			bins[bin/fine][bin%fine] += 1;
		}
		for x in radius+1..w {
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
	let total = now().0-start.0;
	for (id, time) in id![down, forward, reverse] { if time.0 > 5*total/100 { print!("{id}: {:.0}%, ", 100.*time.0 as f64/total as f64); } }
	println!("{}ms", start_time.elapsed().as_millis());
	rank
}

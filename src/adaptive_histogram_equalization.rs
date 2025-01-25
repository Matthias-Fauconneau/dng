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
	struct Histogram { sums: core::simd::u32x64, bins: [core::simd::u32x64; coarse] }
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
		let Histogram{mut sums, mut bins} = Histogram{sums: [0; _].into(), bins: [[0; _].into(); _]};
		for x in -radius..=radius { 
			let ref column = columns[x.max(0) as usize];
			sums += column.sums;
			for segment in 0..coarse { bins[segment] += column.bins[segment]; }
		}
		for x in 0..w-1 {
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
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
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
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

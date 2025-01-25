use {vector::{xy, minmax, MinMax}, image::{Image, XYZ}};
pub fn adaptive_histogram_equalization(image: &Image<impl AsRef<[XYZ<f32>]>>, radius: u32) -> Image<Box<[u32]>> { // TODO: SIMD
	let MinMax{min, max} = minmax(image.data.as_ref().iter().map(|&XYZ{Y,..}| Y)).unwrap();
	const coarse : usize = 0x20;
	const fine : usize = 0x20;
	const Nbins : usize = coarse*fine;
	assert!(image.size.x as usize*(coarse*fine)*2 <= 64*1024*1024);
	let luminance = image.as_ref().map(|XYZ{Y,..}| (f32::ceil((Y-min)/(max-min)*(Nbins as f32)-1.)) as u16);
	let radius = radius as i32;
	struct Histogram { sums: [u32; coarse], bins: [[u32; fine]; coarse] }
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
		let Histogram{mut sums, mut bins} = Histogram{sums: [0; _], bins: [[0; _]; _]};
		for x in -radius..=radius { 
			let ref column = columns[x.max(0) as usize];
			for i in 0..coarse { sums[i] += column.sums[i]; }
			for segment in 0..coarse { for i in 0..fine { bins[segment][i] += column.bins[segment][i]; } }
		}
		for x in 0..w-1 {
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
			// Slide right
			let ref right = columns[(x+radius+1).min(w-1) as usize];
			let ref left = columns[(x-radius).max(0) as usize];
			for i in 0..coarse { sums[i] += right.sums[i] - left.sums[i]; }
			for segment in 0..coarse {
				if right.sums[segment] > 0 { for i in 0..fine { bins[segment][i] += right.bins[segment][i]; } }
				if left.sums[segment] > 0 { for i in 0..fine { bins[segment][i] -= left.bins[segment][i]; } }
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
		let Histogram{mut sums, mut bins} = Histogram{sums: [0; _], bins: [[0; _]; _]};
		for x in (w-1)-radius..=(w-1)+radius {
			let ref column = columns[x.min(w-1) as usize];
			for i in 0..coarse { sums[i] += column.sums[i]; }
			for segment in 0..coarse { for i in 0..fine { bins[segment][i] += column.bins[segment][i]; } }
		}
		for x in (1..w).into_iter().rev() {
			let index = (y*stride+x) as usize;
			let bin = luminance[index] as usize;
			rank[index] = sums[0..bin/fine].iter().sum::<u32>()+bins[bin/fine][0..=bin%fine].iter().sum::<u32>();
			// Slide left
			let ref left = columns[(x-radius-1).max(0) as usize];
			let ref right = columns[(x+radius).min(w-1) as usize];
			for i in 0..coarse { sums[i] += left.sums[i] - right.sums[i]; }
			for segment in 0..coarse {
				if left.sums[segment] > 0 { for i in 0..fine { bins[segment][i] += left.bins[segment][i]; } }
				if right.sums[segment] > 0 { for i in 0..fine { bins[segment][i] -= right.bins[segment][i]; } }
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

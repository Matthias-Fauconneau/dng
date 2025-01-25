use {vector::{xy, minmax, MinMax}, image::{Image, XYZ}};

pub fn adaptive_histogram_equalization(image: &Image<impl AsRef<[XYZ<f32>]>>, radius: u32) -> Image<Box<[u32]>> {
	let MinMax{min, max} = minmax(image.data.as_ref().iter().map(|&XYZ{Y,..}| Y)).unwrap();
	let bins = 0x800; // TODO: multilevel histogram, SIMD
	assert!(image.size.x as usize*bins as usize*2 <= 64*1024*1024);
	let luminance = image.as_ref().map(|XYZ{Y,..}| (f32::ceil((Y-min)/(max-min)*(bins as f32)-1.)) as u16);
	assert!(733 <= radius && radius <= 1024);
	assert!((radius+1+radius) as usize >= bins, "{} {bins}", radius+1+radius);
	let radius = radius as i32;
	let mut column_histograms = vec![vec![0; bins as usize]; luminance.size.x as usize]; // ~120M. More efficient to slide : packed add vs indirect scatter add
	let mut rank = Image::zero(luminance.size);
	let ([w, h], stride) = (luminance.size.signed().into(), luminance.stride as i32);
	assert_eq!(rank.stride as i32, stride);
	for y in -radius..=radius { for x in 0..luminance.size.x { column_histograms[x as usize][luminance[xy{x,y: y.max(0) as u32}] as usize] += 1; } }
	let start = std::time::Instant::now();
	let mut y = 0;
	loop {
		if !(y < h) { break; }
		println!("{y}");
		let mut histogram = vec![0; bins as usize];
		for x in -radius..=radius { for bin in 0..bins { histogram[bin as usize] += column_histograms[x.max(0) as usize][bin as usize]; } }
		for x in 0..w-1 {
			let luminance = luminance[(y*stride+x) as usize];
			rank[(y*stride+x) as usize] = histogram[0..=luminance as usize].iter().sum();
			// Slide right
			for bin in 0..bins { histogram[bin as usize] = (histogram[bin as usize] as i32
				+ column_histograms[(x+radius+1).min(w-1) as usize][bin as usize] as i32
				- column_histograms[(x-radius).max(0) as usize][bin as usize] as i32) as u32;
			}
		}
		{ // Last of row iteration (not sliding further right after)
			let x = w-1;
			let luminance = luminance[(y*stride+x) as usize];
			rank[(y*stride+x) as usize] = histogram[0..=luminance as usize].iter().sum();
		}
		// Slide down
		for x in 0..w {
			column_histograms[x as usize][luminance[((y-radius).max(0)*stride+x) as usize] as usize] -= 1;
			column_histograms[x as usize][luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize] += 1;
		}
		y += 1;
		if !(y < h) { break; }
		let mut histogram = vec![0; bins as usize];
		for x in (w-1)-radius..=(w-1)+radius { for bin in 0..bins { histogram[bin as usize] += column_histograms[x.min(w-1) as usize][bin as usize]; } }
		for x in (1..w).into_iter().rev() {
			let luminance = luminance[(y*stride+x) as usize];
			rank[(y*stride+x) as usize] = histogram[0..=luminance as usize].iter().sum();
			// Slide left
			for bin in 0..bins { histogram[bin as usize] = (histogram[bin as usize] as i32
				+ column_histograms[(x-radius-1).max(0) as usize][bin as usize] as i32
				- column_histograms[(x+radius).min(w-1) as usize][bin as usize] as i32) as u32;
			}
		}
		{ // Back to first of row iteration (not sliding further left after)
			let x = 0;
			let luminance = luminance[(y*stride+x) as usize];
			rank[(y*stride+x) as usize] = histogram[0..=luminance as usize].iter().sum();
		}
		// Slide down
		for x in 0..w {
			column_histograms[x as usize][luminance[((y-radius).max(0)*stride+x) as usize] as usize] -= 1;
			column_histograms[x as usize][luminance[((y+radius+1).min(h-1)*stride+x) as usize] as usize] += 1;
		}
		y += 1;
	}
	println!("{}ms", start.elapsed().as_millis());
	rank
}
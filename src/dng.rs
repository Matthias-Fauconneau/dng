mod gain; use gain::*;
pub fn default<T: Default>() -> T { Default::default() }
use vector::{xy, uint2, vec2, mulv, min, minmax, MinMax};
use image::{Image, XYZ, rgb, rgbf, rgba8, bilinear_sample, blur_3, oetf8_12, sRGB8_OETF12};
use rawler::{rawsource::RawSource, get_decoder, RawImage, RawImageData, Orientation, decoders::{Camera, WellKnownIFD}, formats::tiff::{Entry, Value}};
use rawler::{imgop::xyz::Illuminant::D65, tags::DngTag::OpcodeList2};

pub fn load(path: impl AsRef<std::path::Path>) -> Result<Image<Box<[rgba8]>>, Box<dyn std::error::Error>> {
	let ref file = RawSource::new(path.as_ref())?;
	let decoder = get_decoder(file)?;
	let RawImage{whitelevel, data: RawImageData::Integer(data), width, height, wb_coeffs: rcp_as_shot_neutral, orientation, camera: Camera{forward_matrix, ..}, ..}
		= decoder.raw_image(file, &default(), false)? else {unimplemented!()};
	let tags = decoder.ifd(WellKnownIFD::VirtualDngRawTags)?.unwrap();
	let &[white_level] = whitelevel.0.as_array().unwrap();
	let as_shot_neutral = 1./rgbf::from(*rcp_as_shot_neutral.first_chunk().unwrap()); // FIXME: rawler: expose non-inverted
	let Some(Entry{value: Value::Undefined(code), ..}) = tags.get_entry(OpcodeList2) else {panic!()};
	let gain = gain(code);
	let ref forward = forward_matrix[&D65]; // darktable has a dedicated color temperature (chromatic adaptation) much later in the pipeline
	let forward = *forward.as_array::<{3*3}>().unwrap().map(|v| v.try_into().unwrap()).as_chunks().0.as_array().unwrap();
	let forward = forward.map(rgbf::from)
		.map(|row| row / as_shot_neutral) // ~ * 1/D
		.map(|row| row / white_level as f32)
		.map(<[_;_]>::from);
	let cfa = Image::new(xy{x: width as u32, y: height as u32}, data);
	let scale = vec2::from(gain[0].size-uint2::from(1))/vec2::from(cfa.size/2-uint2::from(1))-vec2::from(f32::EPSILON);
	let image = Image::from_xy(cfa.size/2, |p| {
		let [b, g10, g01, r] = [(0,[0,0]),(1,[1,0]),(2,[0,1]),(3,[1,1])].map(|(i,[x,y])| bilinear_sample(&gain[i], scale*vec2::from(p)) * f32::from(cfa[p*2+xy{x,y}]));
		mulv(forward, [r, (g01+g10)/2., b]).into()
	});
	let pixel_count = image.data.len() as u32;
	let haze = blur_3::<512>(&image);
	let min = min(image.data.iter().zip(&haze.data).map(|(image, haze)| image/haze)).unwrap(); // ~ 1/4
	let image = Image::from_iter(image.size, image.data.iter().zip(haze.data).map(|(image, haze)| image-min*haze)); // Scale haze correction to avoid negative values
	let MinMax{min, max} = minmax(image.data.iter().copied()).unwrap();
	let bins = 0x10000;
	let mut histogram = vec![0; bins];
	for &XYZ{Y,..} in &image.data { histogram[f32::ceil((Y-min.Y)/(max.Y-min.Y)*bins as f32-1.) as usize] += 1; }
	let cdf = histogram.into_iter().scan(0, |a, h| { *a += h; Some(*a) }).collect::<Box<[u32]>>();
	let oetf = &sRGB8_OETF12;
	let image = image.map(|XYZ@XYZ{Y,..}| {
		let rank = cdf[f32::ceil((Y-min.Y)/(max.Y-min.Y)*bins as f32-1.) as usize];
		let f = rank as f32 / (pixel_count-1) as f32;
		if Y > 0. { f*XYZ/Y } else { XYZ }
	});
	//let mean = image.data.iter().copied().sum::<XYZ<f32>>() / image.data.len() as f32;
	let scale = (1./2.)*XYZ::<f32>::from(image.data.len() as f64 / image.data.iter().map(|&XYZ| XYZ::<f64>::from(XYZ)).sum::<XYZ<f64>>());
	const D50 : [[f32; 3]; 3] = [[3.1338561, -1.6168667, -0.4906146], [-0.9787684, 1.9161415, 0.0334540], [0.0719450, -0.2289914, 1.4052427]];
	let image = image.map(|XYZ| rgba8::from(rgb::from(mulv(D50, <[_; _]>::from(scale*XYZ)).map(|c| oetf8_12(oetf, c.clamp(0., 1.))))));
	use Orientation::*;
	let image = match orientation {
		Normal => image,
		Rotate90 => Image::from_xy({let xy{x,y} = image.size; xy{x: y, y: x}}, |xy{x,y}| image[xy{x: y, y: image.size.y-1-x}]),
		o => { eprintln!("{o:#?}"); image },
	};
	Ok(image)
}

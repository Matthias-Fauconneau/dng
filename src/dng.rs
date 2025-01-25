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
	fn xy(XYZ{X,Y,Z}: XYZ<f32>) -> xy<f32> { xy{x: X/(X+Y+Z), y: Y/(X+Y+Z)} }
	let xy = image.as_ref().map(|&XYZ| xy(XYZ));
	let image = Image::from_iter(image.size, image.data.iter().zip(haze.data).map(|(image, haze)| image-min*haze)); // Scale haze correction to avoid negative values
	let MinMax{min, max} = minmax(image.data.iter().copied()).unwrap();
	let bins = 0x10000;
	let mut histogram = vec![0; bins];
	for &XYZ{Y,..} in &image.data { histogram[f32::ceil((Y-min.Y)/(max.Y-min.Y)*bins as f32-1.) as usize] += 1; }
	let cdf = histogram.into_iter().scan(0, |a, h| { *a += h; Some(*a) }).collect::<Box<[u32]>>();
	assert_eq!(cdf[0xFFFF], pixel_count);
	let oetf = &sRGB8_OETF12;
	const D50 : XYZ<f32> = XYZ{X: 0.964212, Y: 1., Z: 0.825188};
	let n = D50;
	//fn f(t: f32) -> f32 { const δ : f32 = 6./29.; if t > δ.powi(3) { 1.16*t.cbrt()-0.16 } else { t/(δ/2.).powi(2) } }
	mod vector_uv { vector::vector!(2 uv T T, u v, U V); } pub use vector_uv::uv;
	fn uv(XYZ{X,Y,Z}: XYZ<f32>) -> uv<f32> { let d = X + 15.*Y + 3.*Z; uv{u: 4.*X/d, v: 9.*Y/d} }
	//dbg!(uv(n));
	//let image = image.map(|XYZ@XYZ{Y,..}| {
	let image = Image::from_iter(image.size, image.data.into_iter().zip(xy.data).map(|(XYZ@XYZ{Y,..},_/*xy{x,y}*/)| {
		let rank = cdf[f32::ceil((Y-min.Y)/(max.Y-min.Y)*bins as f32-1.) as usize];
		let XYZ@XYZ{Y,..} = if Y > 0. { (rank as f32 / pixel_count as f32)*XYZ/Y } else { XYZ };
		assert!(Y >= 0. && Y <= 1.);
		//XYZ{X: f*x, Y: f*y, Z: f*z}

		//let L = f(Y/n.Y);
		let uv_n = uv(n);
		let uv = /*1.3*L*(*/uv(XYZ)-uv_n;//);
		//let uv{u,v} = uv;
		//let C = sqrt(u*u+v*v);
		//let s = C/L;
		//let C = 2.*C;
		let uv = 2.*uv;
		//fn xy(uv{u, v}: uv<f32>) -> xy<f32> { let d = 6.*u - 16.*v + 12.; xy{x: 9.*u/d, y: 4.*v/d} }
		let uv{u,v} = uv_n+uv;
		XYZ{X: (9.*u)/(4.*v)*Y, Y, Z: (12.-3.*u-20.*v)/(4.*v)*Y}
		/*//let D65 = XYZ{X: 0.950489, Y: 1., Z: 1.088840};
		fn f(t: f32) -> f32 { const δ : f32 = 6./29.; if t > δ.powi(3) { t.cbrt() } else { 4./29. + t/(3.*δ.powi(2)) } }
		let L = 1.16*f(Y/n.Y)-0.16;
		let a = 5.*(f(X/n.X)-f(Y/n.Y));
		let b = -2.*(f(Z/n.Z)-f(Y/n.Y));
		let C = sqrt(a*a+b*b);*/
		/*let [dx, dy] = [x-x0, y-y0];
		let r = sqrt(sq(dx)+sq(dy));
		let s = 1./12.;
		let [x, y] = [x0+s*dx/r, y0+s*dy/r];
		let z = 1.-x-y;*/
	}));
	//let mean = image.data.iter().copied().sum::<XYZ<f32>>() / image.data.len() as f32;
	let scale = 1.;//(1./2.)*XYZ::<f32>::from(image.data.len() as f64 / image.data.iter().map(|&XYZ| XYZ::<f64>::from(XYZ)).sum::<XYZ<f64>>());
	const sRGB_from_XYZD50 : [[f32; 3]; 3] = [[3.1338561, -1.6168667, -0.4906146], [-0.9787684, 1.9161415, 0.0334540], [0.0719450, -0.2289914, 1.4052427]];
	let image = image.map(|XYZ| rgba8::from(rgb::from(mulv(sRGB_from_XYZD50, <[_; _]>::from(scale*XYZ)).map(|c| oetf8_12(oetf, c.clamp(0., 1.))))));
	use Orientation::*;
	let image = match orientation {
		Normal => image,
		Rotate90 => Image::from_xy({let xy{x,y} = image.size; xy{x: y, y: x}}, |xy{x,y}| image[xy{x: y, y: image.size.y-1-x}]),
		o => { eprintln!("{o:#?}"); image },
	};
	Ok(image)
}

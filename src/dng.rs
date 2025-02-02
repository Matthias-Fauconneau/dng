pub fn default<T: Default>() -> T { Default::default() }

mod vector_uv { vector::vector!(2 uv T T, u v, U V); } pub use vector_uv::uv;
const fn uv(XYZ{X,Y,Z}: XYZ<f32>) -> uv<f32> { let d = X + 15.*Y + 3.*Z; uv{u: 4.*X/d, v: 9.*Y/d} }
const D50 : XYZ<f32> = XYZ{X: 0.964212, Y: 1., Z: 0.825188};
const uv_D50 : uv<f32> = uv(D50); 
fn to_uv_D50(XYZ: XYZ<f32>) -> uv<f32> { uv(XYZ)-uv_D50 }
fn XYZ(uv{u,v}: uv<f32>, Y: f32) -> XYZ<f32> { XYZ{X: (9.*u)/(4.*v)*Y, Y, Z: (12.-3.*u-20.*v)/(4.*v)*Y} }
fn from_uv_D50(uv: uv<f32>, Y: f32) -> XYZ<f32> { XYZ(uv_D50+uv, Y) }

mod time; use time::Profile;
mod gain; use gain::*;
mod adaptive_histogram_equalization; use adaptive_histogram_equalization::*;

use vector::{xy, uint2, vec2, mat3, mulv, min};
use image::{Image, XYZ, rgb, rgbf, rgba8, bilinear_sample, blur_3, oetf8_12, sRGB8_OETF12};
use rawler::{rawsource::RawSource, get_decoder, RawImage, RawImageData, Orientation, decoders::{Camera, WellKnownIFD}, formats::tiff::{Entry, Value}};
use rawler::{imgop::xyz::Illuminant::D65, tags::DngTag::OpcodeList2};

pub fn load(path: impl AsRef<std::path::Path>) -> Result<Image<Box<[rgba8]>>, Box<dyn std::error::Error>> {
	let mut was = Profile::start();
	let ref file = RawSource::new(path.as_ref())?;
	let decoder = get_decoder(file)?;
	let RawImage{whitelevel, data: RawImageData::Integer(data), width, height, wb_coeffs: rcp_as_shot_neutral, orientation, camera: Camera{forward_matrix, ..}, ..}
		= decoder.raw_image(file, &default(), false)? else {unimplemented!()};
	let cfa = Image::new(xy{x: width as u32, y: height as u32}, data);
	let tags = decoder.ifd(WellKnownIFD::VirtualDngRawTags)?.unwrap();
	let &[white_level] = whitelevel.0.as_array().unwrap();
	let as_shot_neutral = 1./rgbf::from(*rcp_as_shot_neutral.first_chunk().unwrap()); // FIXME: rawler: expose non-inverted
	let ref forward = forward_matrix[&D65]; // darktable has a dedicated color temperature (chromatic adaptation) much later in the pipeline
	let forward = *forward.as_array::<{3*3}>().unwrap().map(|v| v.try_into().unwrap()).as_chunks().0.as_array().unwrap();
	let forward = forward.map(rgbf::from)
		.map(|row| row / as_shot_neutral) // ~ * 1/D
		.map(|row| row / white_level as f32)
		.map(<[_;_]>::from);
	let Some(Entry{value: Value::Undefined(code), ..}) = tags.get_entry(OpcodeList2) else {panic!()};
	let gain = gain(code);
	let gain_map_scale = vec2::from(gain[0].size-uint2::from(1))/vec2::from(cfa.size/2-uint2::from(1))-vec2::from(f32::EPSILON);
	was("decode");
	// Demosaic
	mod vector_BGGR { vector::vector!(4 BGGR T T T T, b g10 g01 r, B G10 G01 R); } pub use vector_BGGR::BGGR;
	let bggr = Image::from_xy(cfa.size/2, |p|
		BGGR{b: (0,[0,0]), g10: (1,[1,0]), g01: (2,[0,1]), r: (3,[1,1])}.map(|(i,[x,y])| bilinear_sample(&gain[i], gain_map_scale*vec2::from(p)) * f32::from(cfa[p*2+xy{x,y}]))
	);
	let mut target = Image::uninitialized(cfa.size);
	for y in 1..target.size.y/2-1 {
		for x in 1..target.size.x/2-1 {
			let bggr = |dx,dy| bggr[xy{x:x-1+dx,y:y-1+dy}];
			let ref mut target = target.slice_mut(2*xy{x,y}, xy{x: 2, y: 2});
			let b = |x,y| { assert_eq!(x%2,0); assert_eq!(y%2,0); bggr((2+x)/2,(2+y)/2).b };
			let r = |x,y| { assert_eq!(x%2,1); assert_eq!(y%2,1); bggr((2+x)/2,(2+y)/2).r };
			let g = |x,y| { let BGGR{g10,g01,..} = bggr((2+x)/2,(2+y)/2); match [x%2, y%2] { [1,0] => g10, [0,1] => g01, _ => unreachable!() } };
			trait F = Fn(u32,u32)->f32;
			fn rgb(target: &mut Image<&mut [XYZ<f32>]>, forward: mat3, x: u32, y: u32, r: impl F, g: impl F, b: impl F) {
				target[xy{x,y}] = mulv(forward, [r(x,y), g(x,y), b(x,y)]).into();
			}
			fn horizontal(c: impl F) -> impl F { move |x,y| (c(x-1,y)+c(x+1,y))/2. }
			fn vertical(c: impl F) -> impl F { move |x,y| (c(x,y-1)+c(x,y+1))/2. }
			fn cardinal(c: impl F) -> impl F { move |x,y| (c(x,y-1)+c(x-1,y)+c(x+1,y)+c(x,y+1))/4. }
			fn diagonal(c: impl F) -> impl F { move |x,y| (c(x-1,y-1)+c(x+1,y-1)+c(x-1,y+1)+c(x+1,y+1))/4. }
			rgb(target, forward, 0,0, diagonal(r), cardinal(g), b);
			rgb(target, forward, 1,0, vertical(r), g, horizontal(b));
			rgb(target, forward, 0,1, horizontal(r), g, vertical(b));
			rgb(target, forward, 1,1, r, cardinal(g), diagonal(b));
		}
	}
	was("demosaic");
	// Dehaze
	let haze = blur_3::<1024>(&target);
	let min = if true { min(target.data.iter().zip(&haze.data).map(|(image, haze)| image/haze)).unwrap().into_iter().min_by(f32::total_cmp).unwrap() } else { 0. }; // ~ 1/4
	target.mut_zip_map(&haze, |image, &haze| image-min*haze); // Scales haze correction to avoid negative values
	was("dehaze");
	if true { // Adaptive Histogram Equalization
		let radius = (std::cmp::max(target.size.x, target.size.y) - 1) / 2;
		let contrast_limited = contrast_limited_adaptive_histogram_equalization(&target, radius);
		assert_eq!(target.stride, contrast_limited.stride);
		target.mut_zip_map(&contrast_limited, |XYZ@XYZ{Y,..}, &contrast_limited| {
			let L = contrast_limited as f32 / (radius+1+radius).pow(2) as f32;
			assert!(L >= 0. && L <= 1.);
			let XYZ@XYZ{Y,..} = if Y > 0. { L*XYZ/Y } else { XYZ };
			assert!(Y >= 0. && Y <= 1.);
			//let XYZ@XYZ{Y,..} = 2.*XYZ;
			from_uv_D50(2.*to_uv_D50(XYZ), Y)
		});
		was("equalization");
	}
	let image = target;
	// rotate map from XYZ to linear sRGB
	const sRGB_from_XYZD50 : [[f32; 3]; 3] = [[3.1338561, -1.6168667, -0.4906146], [-0.9787684, 1.9161415, 0.0334540], [0.0719450, -0.2289914, 1.4052427]];
	let image = image.map(|XYZ| rgb::from(mulv(sRGB_from_XYZD50, <[_; _]>::from(XYZ))));
	was("map");
	// Export
	let jpeg_xl = JPEG_XL::encoder_builder().orientation(u32::from(orientation.to_u16()).try_into()?).speed(1).build()?
		.encode::<_,f32>(&bytemuck::cast_slice::<_,f32>(&image.data), image.size.x, image.size.y)?;
	was("export");
	std::fs::write("../sRGB.linear.jxl", &*jpeg_xl)?;
	was("write");
	// OETF
	let oetf = &sRGB8_OETF12;
	let image = image.map(|rgb| rgba8::from(rgb.map(|c:f32| oetf8_12(oetf, c.clamp(0., 1.)))));
	was("OETF");
	use Orientation::*;
	let image = match orientation {
		Normal => image,
		Rotate90 => Image::from_xy({let xy{x,y} = image.size; xy{x: y, y: x}}, |xy{x,y}| image[xy{x: y, y: image.size.y-1-x}]),
		o => { eprintln!("{o:#?}"); image },
	};
	was("orientation");
	was.print_profile();
	Ok(image)
}

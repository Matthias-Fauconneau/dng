pub fn default<T: Default>() -> T { Default::default() }
struct Profile {
	start: std::time::Instant,
	last: std::time::Instant,
	profile: Vec<(&'static str, std::time::Duration)>,
}
impl Profile {
	pub fn start() -> Self { let start = std::time::Instant::now(); Self{start, last: start, profile: vec![]} }
	pub fn fence(&mut self, was: &'static str) { let now = std::time::Instant::now(); self.profile.push((was, now-self.last)); self.last = now; }
	pub fn print_profile(&self) {
		let total = self.last-self.start;
		for &(id, time) in &self.profile { if time > 5*total/100 { print!("{id}: {:.0}%, ", 100.*time.as_secs_f64()/total.as_secs_f64()); } }
		println!(", {}ms", total.as_millis());
	}
}
impl FnOnce<(&'static str,)> for Profile { type Output = (); extern "rust-call" fn call_once(mut self, (was,): (&'static str,)) -> Self::Output { self.fence(was) }  }
impl FnMut<(&'static str,)> for Profile { extern "rust-call" fn call_mut(&mut self, (was,): (&'static str,)) -> Self::Output { self.fence(was) } }

mod vector_uv { vector::vector!(2 uv T T, u v, U V); } pub use vector_uv::uv;
const fn uv(XYZ{X,Y,Z}: XYZ<f32>) -> uv<f32> { let d = X + 15.*Y + 3.*Z; uv{u: 4.*X/d, v: 9.*Y/d} }
const D50 : XYZ<f32> = XYZ{X: 0.964212, Y: 1., Z: 0.825188};
const uv_D50 : uv<f32> = uv(D50); 
fn to_uv_D50(XYZ: XYZ<f32>) -> uv<f32> { uv(XYZ)-uv_D50 }
fn XYZ(uv{u,v}: uv<f32>, Y: f32) -> XYZ<f32> { XYZ{X: (9.*u)/(4.*v)*Y, Y, Z: (12.-3.*u-20.*v)/(4.*v)*Y} }
fn from_uv_D50(uv: uv<f32>, Y: f32) -> XYZ<f32> { XYZ(uv_D50+uv, Y) }

mod gain; use gain::*;
mod adaptive_histogram_equalization; use adaptive_histogram_equalization::*;

use vector::{xy, uint2, vec2, mulv, min};
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
	was("decode");
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
	// Demosaic
	let mut image = Image::from_xy(cfa.size/2, |p| {
		let [b, g10, g01, r] = [(0,[0,0]),(1,[1,0]),(2,[0,1]),(3,[1,1])].map(|(i,[x,y])| bilinear_sample(&gain[i], gain_map_scale*vec2::from(p)) * f32::from(cfa[p*2+xy{x,y}]));
		mulv(forward, [r, (g01+g10)/2., b]).into()
	});
	was("demosaic");
	// Dehaze
	let haze = blur_3::<512>(&image);
	let min = min(image.data.iter().zip(&haze.data).map(|(image, haze)| image/haze)).unwrap(); // ~ 1/4
	image.mut_zip_map(&haze, |image, haze| image-min*haze); // Scales haze correction to avoid negative values
	was("dehaze");
	if true { // Adaptive Histogram Equalization
		let radius = 486; //(std::cmp::min(image.size.x, image.size.y) - 1) / 2;
		let contrast_limited = contrast_limited_adaptive_histogram_equalization(&image, radius);
		assert_eq!(image.stride, contrast_limited.stride);
		image.mut_zip_map(&contrast_limited, |XYZ@XYZ{Y,..}, &contrast_limited| {
			let L = contrast_limited as f32 / (radius+1+radius).pow(2) as f32;
			assert!(L >= 0. && L <= 1.);
			let XYZ@XYZ{Y,..} = if Y > 0. { L.powi(3)*XYZ/Y } else { XYZ };
			assert!(Y >= 0. && Y <= 1.);
			//let XYZ@XYZ{Y,..} = XYZ;
			from_uv_D50(2.*to_uv_D50(XYZ), Y)
		});
		was("equalization");
	}
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

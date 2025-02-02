pub struct Profile {
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

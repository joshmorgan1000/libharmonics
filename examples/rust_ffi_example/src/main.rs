use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
pub struct HarmonicGraph {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CycleRuntime {
    _private: [u8; 0],
}

#[link(name = "harmonics_ffi")]
extern "C" {
    fn harmonics_parse_graph(src: *const c_char) -> *mut HarmonicGraph;
    fn harmonics_destroy_graph(g: *mut HarmonicGraph);
    fn harmonics_create_csv_producer(path: *const c_char) -> *mut Producer;
    fn harmonics_destroy_producer(p: *mut Producer);
    fn harmonics_bind_producer(g: *mut HarmonicGraph, name: *const c_char, p: *mut Producer);
    fn harmonics_fit(g: *mut HarmonicGraph, epochs: usize);
}

fn main() {
    let graph_src = CString::new("producer d{1}; consumer c; cycle{ d -> c; }").unwrap();
    let csv_path = CString::new("train.csv").unwrap();
    let input = CString::new("d").unwrap();
    unsafe {
        let graph = harmonics_parse_graph(graph_src.as_ptr());
        let data = harmonics_create_csv_producer(csv_path.as_ptr());
        harmonics_bind_producer(graph, input.as_ptr(), data);
        harmonics_fit(graph, 5);
        harmonics_destroy_producer(data);
        harmonics_destroy_graph(graph);
    }
}

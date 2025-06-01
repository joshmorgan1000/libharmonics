use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
pub struct HarmonicGraph {
    _private: [u8; 0],
}
#[repr(C)]
pub struct DistributedScheduler {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Producer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CycleRuntime {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum harmonics_backend_t {
    HARMONICS_BACKEND_CPU = 0,
    HARMONICS_BACKEND_GPU = 1,
    HARMONICS_BACKEND_FPGA = 2,
    HARMONICS_BACKEND_WASM = 3,
    HARMONICS_BACKEND_AUTO = 4,
}

#[link(name = "harmonics_ffi")]
extern "C" {
    fn harmonics_parse_graph(src: *const c_char) -> *mut HarmonicGraph;
    fn harmonics_destroy_graph(g: *mut HarmonicGraph);
    fn harmonics_auto_partition(
        g: *const HarmonicGraph,
        backends: *const harmonics_backend_t,
        count: usize,
    ) -> *mut *mut HarmonicGraph;
    fn harmonics_destroy_partitions(parts: *mut *mut HarmonicGraph, count: usize);
    fn harmonics_create_distributed_scheduler(
        parts: *mut *mut HarmonicGraph,
        count: usize,
        backends: *const harmonics_backend_t,
        secure: bool,
    ) -> *mut DistributedScheduler;
    fn harmonics_destroy_distributed_scheduler(s: *mut DistributedScheduler);
    fn harmonics_scheduler_bind_producer(
        s: *mut DistributedScheduler,
        part: usize,
        name: *const c_char,
        p: *mut Producer,
    );
    fn harmonics_scheduler_fit(s: *mut DistributedScheduler, epochs: usize);
}

fn main() {
    let src = CString::new(
        "producer p{1}; consumer c{1}; layer l1; layer l2; cycle{ p -(id)-> l1 -(id)-> l2 -> c; }",
    )
    .unwrap();
    unsafe {
        let g = harmonics_parse_graph(src.as_ptr());
        let backends = [harmonics_backend_t::HARMONICS_BACKEND_CPU, harmonics_backend_t::HARMONICS_BACKEND_CPU];
        let parts = harmonics_auto_partition(g, backends.as_ptr(), backends.len());
        let sched = harmonics_create_distributed_scheduler(parts, backends.len(), backends.as_ptr(), false);
        harmonics_destroy_partitions(parts, backends.len());
        let dummy_path = CString::new("train.csv").unwrap();
        // Producer creation omitted for brevity
        harmonics_scheduler_fit(sched, 1);
        harmonics_destroy_distributed_scheduler(sched);
        harmonics_destroy_graph(g);
    }
}

#include <harmonics/quantum_backend.hpp>
#include <harmonics/quantum_hardware.hpp>

extern "C" harmonics::QuantumResult harmonics_quantum_execute(const harmonics::QuantumCircuit& qc) {
    (void)harmonics::quantum_device_index();
    return harmonics::simulate(qc);
}

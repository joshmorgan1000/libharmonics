#include <harmonics/quantum_backend.hpp>

extern "C" harmonics::QuantumResult harmonics_quantum_execute(const harmonics::QuantumCircuit& qc) {
    return harmonics::simulate(qc);
}

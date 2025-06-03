#pragma once

#include "harmonics/flight_io.hpp"
#include "harmonics/grpc_io.hpp"
#include "harmonics/net_utils.hpp"
#include "harmonics/spark_dataset.hpp"
#include "harmonics/stream_io.hpp"
#include "harmonics/tcp_io.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace harmonics {

/** Create a Producer from a target URL. Supported schemes:
 *  - file:path       -> FileProducer
 *  - socket:fd       -> SocketProducer
 *  - tcp:host:port   -> TcpProducer
 *  - grpc:host:port  -> GrpcProducer
 *  - flight:host:port -> FlightProducer
 */
inline std::shared_ptr<Producer> make_producer(const std::string& target) {
    auto colon = target.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("invalid target: " + target);
    auto scheme = target.substr(0, colon);
    auto rest = target.substr(colon + 1);
    if (scheme == "file") {
        return std::make_shared<FileProducer>(rest);
    } else if (scheme == "socket") {
        int fd = std::stoi(rest);
        return std::make_shared<SocketProducer>(fd);
    } else if (scheme == "tcp") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid tcp target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<TcpProducer>(host, port);
    } else if (scheme == "grpc") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid grpc target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<GrpcProducer>(host, port);
    } else if (scheme == "flight") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid flight target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<FlightProducer>(host, port);
    } else if (scheme == "spark") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid spark target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<SparkProducer>(host, port);
    }
    throw std::runtime_error("unsupported producer target: " + target);
}

/** Create a Consumer from a target URL. Supported schemes mirror make_producer. */
inline std::shared_ptr<Consumer> make_consumer(const std::string& target) {
    auto colon = target.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("invalid target: " + target);
    auto scheme = target.substr(0, colon);
    auto rest = target.substr(colon + 1);
    if (scheme == "file") {
        return std::make_shared<FileConsumer>(rest);
    } else if (scheme == "socket") {
        int fd = std::stoi(rest);
        return std::make_shared<SocketConsumer>(fd);
    } else if (scheme == "tcp") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid tcp target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<TcpConsumer>(host, port);
    } else if (scheme == "grpc") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid grpc target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<GrpcConsumer>(host, port);
    } else if (scheme == "flight") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid flight target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<FlightConsumer>(host, port);
    } else if (scheme == "spark") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid spark target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<SparkConsumer>(host, port);
    }
    throw std::runtime_error("unsupported consumer target: " + target);
}

/** Create a proof-enabled Producer from a target URL. Supported schemes:
 *  - socket:fd       -> ProofSocketProducer
 *  - tcp:host:port   -> ProofTcpProducer
 */
inline std::shared_ptr<Producer> make_proof_producer(const std::string& target) {
    auto colon = target.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("invalid target: " + target);
    auto scheme = target.substr(0, colon);
    auto rest = target.substr(colon + 1);
    if (scheme == "socket") {
        int fd = std::stoi(rest);
        return std::make_shared<ProofSocketProducer>(fd);
    } else if (scheme == "tcp") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid tcp target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<ProofTcpProducer>(host, port);
    }
    throw std::runtime_error("unsupported proof producer target: " + target);
}

/** Create a proof-enabled Consumer from a target URL. Supported schemes mirror
 * make_proof_producer.
 */
inline std::shared_ptr<Consumer> make_proof_consumer(const std::string& target) {
    auto colon = target.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("invalid target: " + target);
    auto scheme = target.substr(0, colon);
    auto rest = target.substr(colon + 1);
    if (scheme == "socket") {
        int fd = std::stoi(rest);
        return std::make_shared<ProofSocketConsumer>(fd);
    } else if (scheme == "tcp") {
        auto pos = rest.find(':');
        if (pos == std::string::npos)
            throw std::runtime_error("invalid tcp target: " + target);
        auto host = rest.substr(0, pos);
        unsigned short port = static_cast<unsigned short>(std::stoi(rest.substr(pos + 1)));
        return std::make_shared<ProofTcpConsumer>(host, port);
    }
    throw std::runtime_error("unsupported proof consumer target: " + target);
}

} // namespace harmonics

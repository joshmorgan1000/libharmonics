#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "harmonics/tokenizer.hpp"

namespace harmonics {

/** Ratio reference used when propagating widths. */
struct Ratio {
    int lhs{0};
    int rhs{0};
    std::string ref{};
};

/** AST node describing a producer declaration. */
struct ProducerDecl {
    std::string name{};
    std::optional<int> shape{};
    std::optional<Ratio> ratio{};
};

/** AST node describing a consumer declaration. */
struct ConsumerDecl {
    std::string name{};
    std::optional<int> shape{};
};

/** AST node describing a layer declaration. */
struct LayerDecl {
    std::string name{};
    std::optional<Ratio> ratio{};
};

/** Edge in a flow line optionally invoking a function. */
struct FlowArrow {
    bool backward{false};
    std::optional<std::string> func{};
    std::string target{};
};

/** One line in the cycle describing data flow from a source. */
struct FlowLine {
    std::string source{};
    std::vector<FlowArrow> arrows{};
};

struct CycleAST; // forward declaration

/** Conditional block of flow lines executed when a condition is true. */
struct ConditionalBlock {
    std::string condition{};
    std::unique_ptr<CycleAST> if_branch{};
    std::unique_ptr<CycleAST> else_branch{}; // nullptr if absent

    ConditionalBlock() = default;
    ConditionalBlock(const ConditionalBlock& other) {
        condition = other.condition;
        if (other.if_branch)
            if_branch = std::make_unique<CycleAST>(*other.if_branch);
        if (other.else_branch)
            else_branch = std::make_unique<CycleAST>(*other.else_branch);
    }

    ConditionalBlock& operator=(const ConditionalBlock& other) {
        if (this != &other) {
            condition = other.condition;
            if_branch = other.if_branch ? std::make_unique<CycleAST>(*other.if_branch) : nullptr;
            else_branch =
                other.else_branch ? std::make_unique<CycleAST>(*other.else_branch) : nullptr;
        }
        return *this;
    }

    ConditionalBlock(ConditionalBlock&&) noexcept = default;
    ConditionalBlock& operator=(ConditionalBlock&&) noexcept = default;
};

/** Parsed representation of a cycle block. */
struct CycleAST {
    std::vector<FlowLine> lines{};
    std::vector<ConditionalBlock> conditionals{};

    CycleAST() = default;
    CycleAST(const CycleAST& other) = default;
    CycleAST& operator=(const CycleAST& other) = default;
    CycleAST(CycleAST&&) noexcept = default;
    CycleAST& operator=(CycleAST&&) noexcept = default;
};

/** Top-level declarations parsed from the DSL. */
struct DeclarationsAST {
    std::vector<ProducerDecl> producers{};
    std::vector<ConsumerDecl> consumers{};
    std::vector<LayerDecl> layers{};
    std::optional<CycleAST> cycle{};

    DeclarationsAST() = default;
    DeclarationsAST(const DeclarationsAST& other) = default;
    DeclarationsAST& operator=(const DeclarationsAST& other) = default;
    DeclarationsAST(DeclarationsAST&&) noexcept = default;
    DeclarationsAST& operator=(DeclarationsAST&&) noexcept = default;
};

/** Complete AST for a harmonic definition. */
struct HarmonicAST {
    std::string name{};
    DeclarationsAST decls{};
};

/**
 * Recursive descent parser for the Harmonics DSL.
 * See grammar/harp.peg for the formal grammar specification.
 */
class Parser {
  public:
    /** Construct a parser from the given source string. */
    explicit Parser(std::string_view src) : tok_{src} {}

    /** Parse only the declarations section of the source. */
    DeclarationsAST parse_declarations();
    /** Parse a full harmonic definition including its name. */
    HarmonicAST parse_harmonic();

  private:
    Token peek() { return tok_.peek(); }
    Token next() { return tok_.next(); }
    bool accept(TokenType t);
    Token expect(TokenType t);

    std::optional<int> parse_shape();
    std::optional<Ratio> parse_ratio();

    ProducerDecl parse_producer();
    ConsumerDecl parse_consumer();
    LayerDecl parse_layer();
    FlowLine parse_flow_line(std::string& last_src, bool& have_src);
    CycleAST parse_flow_block();
    ConditionalBlock parse_conditional();
    CycleAST parse_cycle();

    Tokenizer tok_;
};

inline bool Parser::accept(TokenType t) {
    if (peek().type == t) {
        next();
        return true;
    }
    return false;
}

inline Token Parser::expect(TokenType t) {
    auto tok = next();
    if (tok.type != t) {
        std::string msg = "expected " + std::string(tokenTypeName(t)) + " at line " +
                          std::to_string(tok.line) + ", column " + std::to_string(tok.column) +
                          ", got '" + tok.text + "'";
        throw std::runtime_error(msg);
    }
    return tok;
}

inline std::optional<int> Parser::parse_shape() {
    if (accept(TokenType::LBrace)) {
        auto num = expect(TokenType::Number);
        expect(TokenType::RBrace);
        return std::stoi(num.text);
    }
    return std::nullopt;
}

inline std::optional<Ratio> Parser::parse_ratio() {
    if (peek().type == TokenType::Number) {
        auto lhs = expect(TokenType::Number);
        expect(TokenType::Slash);
        auto rhs = expect(TokenType::Number);
        auto ref = expect(TokenType::Identifier);
        return Ratio{std::stoi(lhs.text), std::stoi(rhs.text), ref.text};
    }
    return std::nullopt;
}

inline ProducerDecl Parser::parse_producer() {
    expect(TokenType::Producer);
    auto name = expect(TokenType::Identifier);
    auto shape = parse_shape();
    auto ratio = parse_ratio();
    expect(TokenType::Semicolon);
    return ProducerDecl{name.text, shape, ratio};
}

inline ConsumerDecl Parser::parse_consumer() {
    expect(TokenType::Consumer);
    auto name = expect(TokenType::Identifier);
    auto shape = parse_shape();
    expect(TokenType::Semicolon);
    return ConsumerDecl{name.text, shape};
}

inline LayerDecl Parser::parse_layer() {
    expect(TokenType::Layer);
    auto name = expect(TokenType::Identifier);
    auto ratio = parse_ratio();
    expect(TokenType::Semicolon);
    return LayerDecl{name.text, ratio};
}

inline FlowLine Parser::parse_flow_line(std::string& last_src, bool& have_src) {
    FlowLine line;
    if (peek().type == TokenType::Identifier) {
        auto src = expect(TokenType::Identifier);
        line.source = src.text;
        last_src = line.source;
        have_src = true;
    } else {
        if (!have_src) {
            Token tok = peek();
            std::string msg = "flow line missing source at line " + std::to_string(tok.line) +
                              ", column " + std::to_string(tok.column);
            throw std::runtime_error(msg);
        }
        line.source = last_src;
    }

    while (true) {
        FlowArrow arrow;
        if (accept(TokenType::Arrow)) {
            arrow.backward = false;
        } else if (accept(TokenType::Hyphen)) {
            expect(TokenType::LParen);
            auto fn = expect(TokenType::Identifier);
            expect(TokenType::RParen);
            expect(TokenType::Arrow);
            arrow.backward = false;
            arrow.func = fn.text;
        } else if (accept(TokenType::BackArrow)) {
            arrow.backward = true;
            if (accept(TokenType::LParen)) {
                auto fn = expect(TokenType::Identifier);
                expect(TokenType::RParen);
                expect(TokenType::Hyphen);
                arrow.func = fn.text;
            }
        } else {
            break;
        }

        auto dst = expect(TokenType::Identifier);
        arrow.target = dst.text;
        line.arrows.push_back(arrow);

        if (accept(TokenType::Pipe))
            continue;
        if (peek().type == TokenType::Semicolon)
            break;
    }

    expect(TokenType::Semicolon);
    return line;
}

inline CycleAST Parser::parse_flow_block() {
    CycleAST block;
    expect(TokenType::LBrace);
    std::string last_src;
    bool have_src = false;
    while (!accept(TokenType::RBrace)) {
        if (peek().type == TokenType::If) {
            block.conditionals.push_back(parse_conditional());
        } else {
            block.lines.push_back(parse_flow_line(last_src, have_src));
        }
    }
    return block;
}

inline ConditionalBlock Parser::parse_conditional() {
    ConditionalBlock cond;
    expect(TokenType::If);
    auto name = expect(TokenType::Identifier);
    cond.condition = name.text;
    cond.if_branch = std::make_unique<CycleAST>(parse_flow_block());
    if (accept(TokenType::Else)) {
        cond.else_branch = std::make_unique<CycleAST>(parse_flow_block());
    }
    return cond;
}

inline CycleAST Parser::parse_cycle() {
    expect(TokenType::Cycle);
    return parse_flow_block();
}

inline DeclarationsAST Parser::parse_declarations() {
    DeclarationsAST ast;
    while (!tok_.eof()) {
        auto t = peek();
        switch (t.type) {
        case TokenType::Producer:
            ast.producers.push_back(parse_producer());
            break;
        case TokenType::Consumer:
            ast.consumers.push_back(parse_consumer());
            break;
        case TokenType::Layer:
            ast.layers.push_back(parse_layer());
            break;
        case TokenType::Cycle:
            ast.cycle = parse_cycle();
            break;
        default:
            return ast;
        }
    }
    return ast;
}

inline HarmonicAST Parser::parse_harmonic() {
    expect(TokenType::Harmonic);
    auto name = expect(TokenType::Identifier);
    expect(TokenType::LBrace);
    auto decls = parse_declarations();
    expect(TokenType::RBrace);
    return HarmonicAST{name.text, std::move(decls)};
}

} // namespace harmonics

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace harmonics {

/** All token kinds produced by the DSL tokenizer. */
enum class TokenType {
    End,
    Harmonic,
    Producer,
    Consumer,
    Layer,
    Cycle,
    If,
    Else,
    Identifier,
    Number,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Semicolon,
    Arrow,     // ->
    BackArrow, // <-
    Slash,
    Hyphen,
    Pipe,
};

inline const char* tokenTypeName(TokenType t) {
    switch (t) {
    case TokenType::End:
        return "end of input";
    case TokenType::Harmonic:
        return "'harmonic'";
    case TokenType::Producer:
        return "'producer'";
    case TokenType::Consumer:
        return "'consumer'";
    case TokenType::Layer:
        return "'layer'";
    case TokenType::Cycle:
        return "'cycle'";
    case TokenType::If:
        return "'if'";
    case TokenType::Else:
        return "'else'";
    case TokenType::Identifier:
        return "identifier";
    case TokenType::Number:
        return "number";
    case TokenType::LBrace:
        return "'{'";
    case TokenType::RBrace:
        return "'}'";
    case TokenType::LParen:
        return "'('";
    case TokenType::RParen:
        return "')'";
    case TokenType::Semicolon:
        return "';'";
    case TokenType::Arrow:
        return "'->'";
    case TokenType::BackArrow:
        return "'<-'";
    case TokenType::Slash:
        return "'/'";
    case TokenType::Hyphen:
        return "'-'";
    case TokenType::Pipe:
        return "'|'";
    }
    return "unknown";
}

/** A single token with location information. */
struct Token {
    TokenType type{TokenType::End};
    std::string text{};
    std::size_t line{0};
    std::size_t column{0};
};

/**
 * Lightweight tokenizer for the Harmonics DSL.
 *
 * It performs minimal lexical analysis and keeps track of line and
 * column positions for diagnostics.
 */
class Tokenizer {
  public:
    /** Create a tokenizer over the given source text. */
    explicit Tokenizer(std::string_view src) : src_{src} {}

    Token next();
    Token peek();
    bool eof();

  private:
    char current() const;
    bool match(char c);
    bool match(std::string_view s);
    void skip_whitespace();

    Token identifier();
    Token number();
    Token punctuation();

    std::size_t index_{0};
    std::size_t line_{1};
    std::size_t col_{1};
    std::string_view src_{};
    Token lookahead_{};
    bool has_lookahead_{false};
};

inline char Tokenizer::current() const { return index_ < src_.size() ? src_[index_] : '\0'; }

inline bool Tokenizer::match(char c) {
    if (current() == c) {
        ++index_;
        ++col_;
        return true;
    }
    return false;
}

inline bool Tokenizer::match(std::string_view s) {
    if (src_.substr(index_, s.size()) == s) {
        index_ += s.size();
        col_ += s.size();
        return true;
    }
    return false;
}

inline void Tokenizer::skip_whitespace() {
    while (true) {
        if (match('\n')) {
            ++line_;
            col_ = 1;
        } else if (std::isspace(static_cast<unsigned char>(current()))) {
            match(current());
        } else {
            break;
        }
    }
}

inline Token Tokenizer::identifier() {
    std::size_t start = index_;
    std::size_t start_col = col_;
    while (std::isalnum(static_cast<unsigned char>(current())) || current() == '_') {
        match(current());
    }
    std::string_view word = src_.substr(start, index_ - start);

    Token tok{TokenType::Identifier, std::string(word), line_, start_col};
    if (word == "harmonic")
        tok.type = TokenType::Harmonic;
    else if (word == "producer")
        tok.type = TokenType::Producer;
    else if (word == "consumer")
        tok.type = TokenType::Consumer;
    else if (word == "layer")
        tok.type = TokenType::Layer;
    else if (word == "cycle")
        tok.type = TokenType::Cycle;
    else if (word == "if")
        tok.type = TokenType::If;
    else if (word == "else")
        tok.type = TokenType::Else;

    return tok;
}

inline Token Tokenizer::number() {
    std::size_t start = index_;
    std::size_t start_col = col_;
    while (std::isdigit(static_cast<unsigned char>(current()))) {
        match(current());
    }
    return {TokenType::Number, std::string(src_.substr(start, index_ - start)), line_, start_col};
}

inline Token Tokenizer::punctuation() {
    std::size_t start_col = col_;
    if (match("->")) {
        return {TokenType::Arrow, "->", line_, start_col};
    }
    if (match("<-")) {
        return {TokenType::BackArrow, "<-", line_, start_col};
    }

    char c = current();
    match(c);
    switch (c) {
    case '{':
        return {TokenType::LBrace, "{", line_, start_col};
    case '}':
        return {TokenType::RBrace, "}", line_, start_col};
    case '(':
        return {TokenType::LParen, "(", line_, start_col};
    case ')':
        return {TokenType::RParen, ")", line_, start_col};
    case ';':
        return {TokenType::Semicolon, ";", line_, start_col};
    case '/':
        return {TokenType::Slash, "/", line_, start_col};
    case '-':
        return {TokenType::Hyphen, "-", line_, start_col};
    case '|':
        return {TokenType::Pipe, "|", line_, start_col};
    default:
        throw std::runtime_error("unexpected character '" + std::string(1, c) + "' at line " +
                                 std::to_string(line_) + ", column " + std::to_string(start_col));
    }
}

inline Token Tokenizer::next() {
    if (has_lookahead_) {
        has_lookahead_ = false;
        return lookahead_;
    }

    skip_whitespace();
    std::size_t start_col = col_;

    char c = current();
    if (c == '\0') {
        return {TokenType::End, "", line_, col_};
    } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        return identifier();
    } else if (std::isdigit(static_cast<unsigned char>(c))) {
        return number();
    } else {
        return punctuation();
    }
}

inline Token Tokenizer::peek() {
    if (!has_lookahead_) {
        lookahead_ = next();
        has_lookahead_ = true;
    }
    return lookahead_;
}

inline bool Tokenizer::eof() { return peek().type == TokenType::End; }

inline std::vector<Token> tokenize(std::string_view src) {
    Tokenizer t{src};
    std::vector<Token> tokens;
    while (!t.eof()) {
        tokens.push_back(t.next());
    }
    tokens.push_back(t.next()); // push End token
    return tokens;
}

} // namespace harmonics

#include <gtest/gtest.h>
#include <harmonics/tokenizer.hpp>

using harmonics::Token;
using harmonics::tokenize;
using harmonics::TokenType;

TEST(TokenizerTest, BasicTokens) {
    const char* src = "producer input {28} 1/1;";
    auto toks = tokenize(src);
    EXPECT_EQ(toks.size(), 10u);
    EXPECT_EQ(toks[0].type, TokenType::Producer);
    EXPECT_EQ(toks[1].type, TokenType::Identifier);
    EXPECT_EQ(toks[1].text, "input");
    EXPECT_EQ(toks[2].type, TokenType::LBrace);
    EXPECT_EQ(toks[3].type, TokenType::Number);
    EXPECT_EQ(toks[3].text, "28");
    EXPECT_EQ(toks[4].type, TokenType::RBrace);
    EXPECT_EQ(toks[5].type, TokenType::Number);
    EXPECT_EQ(toks[5].text, "1");
    EXPECT_EQ(toks[6].type, TokenType::Slash);
    EXPECT_EQ(toks[7].type, TokenType::Number);
    EXPECT_EQ(toks[7].text, "1");
    EXPECT_EQ(toks[8].type, TokenType::Semicolon);
    EXPECT_EQ(toks.back().type, TokenType::End);
}

TEST(TokenizerTest, ReservedAsIdentifier) {
    const char* src = "producer producer;";
    auto toks = tokenize(src);
    EXPECT_EQ(toks.size(), 4u);
    EXPECT_EQ(toks[0].type, TokenType::Producer);
    EXPECT_EQ(toks[1].type, TokenType::Producer); // reserved keyword, not identifier
    EXPECT_EQ(toks[2].type, TokenType::Semicolon);
}

TEST(TokenizerTest, FlowTokens) {
    const char* src = "a -(relu)-> b <-(loss)- c;";
    auto toks = tokenize(src);
    EXPECT_EQ(toks.size(), 15u);
    EXPECT_EQ(toks[0].type, TokenType::Identifier);
    EXPECT_EQ(toks[1].type, TokenType::Hyphen);
    EXPECT_EQ(toks[2].type, TokenType::LParen);
    EXPECT_EQ(toks[3].type, TokenType::Identifier);
    EXPECT_EQ(toks[4].type, TokenType::RParen);
    EXPECT_EQ(toks[5].type, TokenType::Arrow);
    EXPECT_EQ(toks[6].type, TokenType::Identifier);
    EXPECT_EQ(toks[7].type, TokenType::BackArrow);
    EXPECT_EQ(toks[8].type, TokenType::LParen);
    EXPECT_EQ(toks[9].type, TokenType::Identifier);
    EXPECT_EQ(toks[10].type, TokenType::RParen);
    EXPECT_EQ(toks[11].type, TokenType::Hyphen);
    EXPECT_EQ(toks[12].type, TokenType::Identifier);
    EXPECT_EQ(toks[13].type, TokenType::Semicolon);
    EXPECT_EQ(toks[14].type, TokenType::End);
}

TEST(TokenizerTest, ConditionalTokens) {
    const char* src = "if cond { a -> b | -> c; } else { a -> b; }";
    auto toks = tokenize(src);
    EXPECT_EQ(toks.empty(), false);
    EXPECT_EQ(toks[0].type, TokenType::If);
    EXPECT_EQ(toks[1].type, TokenType::Identifier);
    bool has_pipe = false;
    for (const auto& t : toks) {
        if (t.type == TokenType::Pipe)
            has_pipe = true;
    }
    EXPECT_EQ(has_pipe, true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

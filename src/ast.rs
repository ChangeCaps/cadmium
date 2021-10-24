use lasagna::*;

#[derive(Named, Clone, Debug, PartialEq, Eq, Hash)]
#[name = "identifier"]
pub struct Ident {
    span: Span,
    pub value: String,
}

impl Spanned for Ident {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

impl AsRef<String> for Ident {
    #[inline]
    fn as_ref(&self) -> &String {
        &self.value
    }
}

impl AsRef<str> for Ident {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl Token for Ident {
    #[inline]
    fn lex(lexer: &mut impl Lexer<Output = char>) -> Result<Self, ParseError> {
        let span = lexer.span(0);
        let mut ident = String::new();

        loop {
            let c = lexer.peek();

            if let Some(c) = c {
                if c.is_alphabetic() || *c == '_' || (!ident.is_empty() && c.is_numeric()) {
                    ident.push(*c);

                    lexer.consume();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if ident.is_empty() {
            Err(ParseError::msg(
                span,
                "identifiers must contain at least one character",
            ))
        } else {
            Ok(Ident {
                span: span | lexer.span(0),
                value: ident,
            })
        }
    }
}

#[derive(Named, Clone, Debug, PartialEq, Eq, Hash)]
#[name = "integer"]
pub struct Integer {
    span: Span,
    pub value: i64,
}

impl Spanned for Integer {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

impl Token for Integer {
    #[inline]
    fn lex(lexer: &mut impl Lexer<Output = char>) -> Result<Self, ParseError> {
        let span = lexer.span(0);
        let radix = 10;
        let mut number = String::new();

        loop {
            let c = lexer.peek();

            if let Some(c) = c {
                if c.is_digit(radix) {
                    number.push(*c);

                    lexer.consume();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if number.is_empty() {
            Err(ParseError::msg(
                span,
                "integers must contain at least one digit",
            ))
        } else {
            Ok(Integer {
                span: span | lexer.span(0),
                value: number.parse().unwrap(),
            })
        }
    }
}

#[derive(Named, Clone, Debug, PartialEq, Eq, Hash)]
#[name = "eof"]
pub struct Eof(Span);

impl Token for Eof {
    #[inline]
    fn lex(lexer: &mut impl Lexer<Output = char>) -> Result<Self, ParseError> {
        if lexer.next().is_none() {
            Ok(Self(lexer.span(0)))
        } else {
            Err(ParseError::msg(lexer.span(0), "expected eof"))
        }
    }
}

impl Spanned for Eof {
    #[inline]
    fn span(&self) -> Span {
        self.0
    }
}

#[derive(Token, Named, Clone, PartialEq, Eq, Hash)]
#[name = "token"]
pub enum CadmiumToken {
    // symbols
    #[token = "{"]
    OpenBrace,
    #[token = "}"]
    CloseBrace,
    #[token = "["]
    OpenBracket,
    #[token = "]"]
    CloseBracket,
    #[token = "("]
    OpenParen,
    #[token = ")"]
    CloseParen,
    #[token = "=="]
    EqualEqual,
    #[token = "!="]
    NotEqual,
    #[token = "||"]
    OrOr,
    #[token = "&&"]
    AndAnd,
    #[token = "="]
    Equal,
    #[token = "+"]
    Plus,
    #[token = "-"]
    Minus,
    #[token = "*"]
    Asterisk,
    #[token = "/"]
    Slash,
    #[token = "!"]
    Not,
    #[token = "|"]
    Or,
    #[token = "&"]
    And,
    #[token = ";"]
    SemiColon,
    #[token = ":"]
    Colon,
    #[token = ","]
    Comma,

    // keywords
    #[token = "let"]
    Let,
    #[token = "if"]
    If,
    #[token = "true"]
    True,
    #[token = "false"]
    False,

    // misc
    #[token]
    Ident(Ident),
    #[token]
    Integer(Integer),
    #[token]
    Eof(Eof),
}

#[derive(Parse, Spanned, Clone, Debug)]
pub struct ParenExpr {
    pub open: OpenParen,
    pub expr: Expr,
    pub close: CloseParen,
}

#[derive(Parse, Spanned, Clone, Debug)]
pub enum Bool {
    True(True),
    False(False),
}

#[derive(Parse, Spanned, Clone, Debug)]
pub enum TermExpr {
    Paren(Box<ParenExpr>),
    Variable(Ident),
    Integer(Integer),
    Bool(Bool),
}

#[derive(Parse, Spanned, Clone, Debug)]
pub enum UnaryExpr {
    Term(TermExpr),
}

#[derive(Parse, Spanned, Clone, Debug)]
pub enum BinaryOp {
    Add(Plus),
    Sub(Minus),
    Mul(Asterisk),
    Div(Slash),
    Equal(EqualEqual),
    NotEqual(NotEqual),
}

impl std::fmt::Display for BinaryOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Add(_) => "+",
                Self::Sub(_) => "-",
                Self::Mul(_) => "*",
                Self::Div(_) => "/",
                Self::Equal(_) => "==",
                Self::NotEqual(_) => "!=",
            }
        )
    }
}

impl BinaryOp {
    #[inline]
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Equal(_) | Self::NotEqual(_) => 0,
            Self::Add(_) | Self::Sub(_) => 1,
            Self::Mul(_) | Self::Div(_) => 2,
        }
    }

    #[inline]
    pub fn is_numeric(&self) -> bool {
        match self {
            Self::Add(_) | Self::Sub(_) | Self::Mul(_) | Self::Div(_) => true,
            _ => false,
        }
    }
}

#[derive(Spanned, Clone, Debug)]
pub struct BinaryExpr {
    pub lhs: Expr,
    pub op: BinaryOp,
    pub rhs: Expr,
}

impl Parse for Expr {
    type Source = CadmiumToken;

    #[inline]
    fn parse(parser: &mut impl Parser<Source = Self::Source>) -> Result<Self, ParseError> {
        let lhs = parser.parse::<UnaryExpr>()?;

        if let Some(op) = parser.try_parse::<BinaryOp>() {
            let rhs = parser.parse::<Expr>()?;

            if let Expr::Binary(rhs_bin) = rhs {
                if rhs_bin.op.precedence() > op.precedence() {
                    Ok(Expr::Binary(Box::new(BinaryExpr {
                        lhs: Expr::Unary(lhs),
                        op,
                        rhs: Expr::Binary(rhs_bin),
                    })))
                } else {
                    Ok(Expr::Binary(Box::new(BinaryExpr {
                        lhs: Expr::Binary(Box::new(BinaryExpr {
                            lhs: Expr::Unary(lhs),
                            op,
                            rhs: rhs_bin.lhs,
                        })),
                        op: rhs_bin.op,
                        rhs: rhs_bin.rhs,
                    })))
                }
            } else {
                Ok(Expr::Binary(Box::new(BinaryExpr {
                    lhs: Expr::Unary(lhs),
                    op,
                    rhs,
                })))
            }
        } else {
            Ok(Expr::Unary(lhs))
        }
    }
}

#[derive(Spanned, Clone, Debug)]
pub enum Expr {
    Binary(Box<BinaryExpr>),
    Unary(UnaryExpr),
}

#[derive(Parse, Spanned, Clone, Debug)]
pub struct BlockStmt {
    pub open: OpenBrace,
    pub stmts: VecTerminated<Stmt, CloseBrace>,
    pub close: CloseBrace,
}

#[derive(Parse, Spanned, Clone, Debug)]
pub struct IfStmt {
    pub if_: If,
    pub expr: Expr,
    pub block: BlockStmt,
}

#[derive(Parse, Spanned, Clone, Debug)]
pub struct LetStmt {
    pub let_: Let,
    pub ident: Ident,
    pub equal: Equal,
    pub expr: Expr,
    pub semi_colon: SemiColon,
}

#[derive(Parse, Spanned, Clone, Debug)]
pub struct AssignStmt {
    pub lhs: Expr,
    pub equal: Equal,
    pub rhs: Expr,
    pub semi_colon: SemiColon,
}

#[derive(Parse, Spanned, Clone, Debug)]
pub enum Stmt {
    #[parse(peek = Let)]
    Let(LetStmt),
    Assign(AssignStmt),
    #[parse(peek = If)]
    If(IfStmt),
}

/*
#[derive(Parse, Clone, Debug)]
pub enum Decl {}
*/

#[derive(Parse, Clone, Debug)]
pub struct AstProgram {
    pub stmts: VecTerminated<Stmt, Eof>,
}

impl AstProgram {
    #[inline]
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        let mut parser = SkipWhitespace::new(CharsLexer::new(source.chars())).parse_as();

        parser.parse()
    }
}

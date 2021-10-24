#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    I32,
    F32,
    Bool,
    Void,
    Ref(Box<Type>),
}

impl std::fmt::Display for Type {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::F32 => write!(f, "f32"),
            Self::Bool => write!(f, "bool"),
            Self::Void => write!(f, "void"),
            Self::Ref(inner) => write!(f, "&{}", inner),
        }
    }
}

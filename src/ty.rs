#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    I32,
    F32,
    Bool,
    Void,
    Ptr(Box<Type>),
    Fn(Vec<Type>, Box<Type>),
}

impl std::fmt::Display for Type {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::F32 => write!(f, "f32"),
            Self::Bool => write!(f, "bool"),
            Self::Void => write!(f, "void"),
            Self::Fn(args, rt) => {
                write!(f, "fn(")?;

                for (i, arg) in args.iter().enumerate() {
                    if i == args.len() - 1 {
                        write!(f, "{}, ", arg)?;
                    } else {
                        write!(f, "{}", arg)?;
                    }
                }

                write!(f, ") -> {}", rt)
            }
            Self::Ptr(inner) => write!(f, "*{}", inner),
        }
    }
}
